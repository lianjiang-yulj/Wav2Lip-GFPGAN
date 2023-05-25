import queue
from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
import time
import copy

import gc

from aiortc.contrib.media import MediaPlayer
from aiortc.mediastreams import MediaStreamError
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import asyncio
import nest_asyncio

from fractions import Fraction

nest_asyncio.apply()

class RtWav2Lip(object):
    def __init__(self, argv, audio_mq = None, video_mq = None):

        self.parser = argparse.ArgumentParser(
            description='Inference code to lip-sync videos in the wild using Wav2Lip models')

        self.parser.add_argument('--checkpoint_path', type=str,
                            help='Name of saved checkpoint to load weights from', required=True)

        self.parser.add_argument('--face', type=str,
                            help='Filepath of video/image that contains faces to use', required=True)
        self.parser.add_argument('--audio', type=str,
                            help='Filepath of video/audio file to use as raw audio source', required=True)
        self.parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                            default='results/result_voice.mp4')

        self.parser.add_argument('--static', type=bool,
                            help='If True, then use only first video frame for inference', default=False)
        self.parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                            default=25., required=False)

        self.parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                            help='Padding (top, bottom, left, right). Please adjust to include chin at least')

        self.parser.add_argument('--face_det_batch_size', type=int,
                            help='Batch size for face detection', default=16)
        self.parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

        self.parser.add_argument('--resize_factor', default=1, type=int,
                            help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

        self.parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                            help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                                 'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

        self.parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                            help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                                 'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

        self.parser.add_argument('--rotate', default=False, action='store_true',
                            help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                                 'Use if you get a flipped result, despite feeding a normal looking video')

        self.parser.add_argument('--nosmooth', default=False, action='store_true',
                            help='Prevent smoothing face detections over a short temporal window')

        self.args = self.parser.parse_args(args = argv)



        self.args.img_size = 96

        self.mel_step_size = 16
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} for inference.'.format(self.device))

        if os.path.isfile(self.args.face) and self.args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            self.args.static = True

        self.audio_mq = audio_mq
        self.video_mq = video_mq

        self.is_pause = False
        self.is_stop = False

        print('args content: ',self.args.__dict__)

        self.g_save_detect_faces = True
        self.g_save_masked_faces = True
        self.g_save_pred_faces = True
        self.g_save_pred_resize_and_connect_faces = True
        return

    async def my_track_consume(self, track):
        if track is None:
            return None

        frame = None
        try:
            frame = await track.recv()
        except MediaStreamError as mse:
            # print(f'!!!!!!!!!!!!!ERROR: recieve MediaStreamError, {track}', mse)
            return None
        except Exception as e:
            # print(f'!!!!!!!!!!!!!ERROR: recieve Exception, {track}', e)
            return None
        return frame

    async def get_audio_frames(self, track):
        frame = await self.my_track_consume(track)

        # # Convert to float32 numpy array
        # floatArray = frame
        floatArray = None
        if frame is not None: floatArray = frame.to_ndarray(format='s16', layout='stereo')

        # Put these samples into the mic queue
        return frame, floatArray


    def set_pause(self, p):
        self.is_pause = p

    def get_pause(self):
        return self.is_pause

    def set_stop(self, s):
        self.is_stop = s

    def get_stop(self):
        return self.is_stop

    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self, images):
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                                flip_input=False, device=self.device)

        batch_size = self.args.face_det_batch_size

        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError(
                        'Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.args.pads #可以通过c
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.args.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        del detector
        return results

    def datagen(self, frames, mels, face_det_results):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.args.box[0] == -1:
            if not self.args.static:
                pass
                #face_det_results = self.face_detect(frames)  # BGR2RGB for CNN face detection
            else:
                pass
                #face_det_results = self.face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.args.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        to_print = True
        batch_print = True
        for i, m in enumerate(mels):
            idx = 0 if self.args.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            if to_print : print(f'->>>>>>>>>>>>>>>> face detect before size:', face.shape)
            face = cv2.resize(face, (self.args.img_size, self.args.img_size))

            if to_print: print(f'->>>>>>>>>>>>>>>> face detect after resize:', self.args.img_size)
            to_print = False

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.args.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.args.img_size // 2:] = 0

                if batch_print: self.save_masked_faces(img_masked)
                batch_print = False

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.args.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch



    def _load(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def load_model(self, path):
        model = Wav2Lip()
        print("Load checkpoint from: {}".format(path))
        checkpoint = self._load(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()

    def main(self):
        if not os.path.isfile(self.args.face):
            raise ValueError('--face argument must be a valid path to video/image file')

        elif self.args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            self.full_frames = [cv2.imread(self.args.face)]
            self.fps = self.args.fps

        else:
            video_stream = cv2.VideoCapture(self.args.face)
            self.fps = video_stream.get(cv2.CAP_PROP_FPS)

            print('->>>>>>>>>>>>>>>> Reading video frames..., fps: ', self.fps)

            self.full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if self.args.resize_factor > 1:
                    frame = cv2.resize(frame,
                                       (frame.shape[1] // self.args.resize_factor, frame.shape[0] // self.args.resize_factor))

                if self.args.rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = self.args.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]

                self.full_frames.append(frame)

        print("->>>>>>>>>>>>>>>> Number of frames available for inference: " + str(len(self.full_frames)))

        if not self.args.audio.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(self.args.audio, 'temp/temp.wav')

            subprocess.call(command, shell=True)
            self.args.audio = 'temp/temp.wav'

        # 提前检测
        self.face_det_results = self.face_detect(self.full_frames)  # BGR2RGB for CNN face detection
        print("->>>>>>>>>>>>>>>> Length of face_det_results: {}".format(len(self.face_det_results)))

        self.save_detect_faces(self.face_det_results)

        self.model = self.load_model(self.args.checkpoint_path)
        print("->>>>>>>>>>>>>>>> Model loaded")
        gc.enable()
        self.start_time_sec = int(time.time())
        self.frameNumberTotal = 0
        self.fps_fraction =Fraction(50, int(self.fps))
        self.numerator = self.fps_fraction.numerator
        self.denominator = self.fps_fraction.denominator




    def inference(self):

        infer_start = int(time.time())
        print(f"->>>>>>>>>>>>>>>> inference prepare start at {infer_start}")

        myplayer = MediaPlayer(file=self.args.audio, loop=False)
        print(f"->>>>>>>>>>>>>>>> Audio loaded, {self.args.audio}")

        wav = audio.load_wav(self.args.audio, 16000)
        mel = audio.melspectrogram(wav)
        print(mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')



        frame_h, frame_w = self.full_frames[0].shape[:-1]
        # out = cv2.VideoWriter('temp/result.avi',
        #                       cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
        mel_chunks = []
        mel_idx_multiplier = 80. / self.fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1

        print("->>>>>>>>>>>>>>>> Length of mel chunks: {}".format(len(mel_chunks)))
        full_frames = self.full_frames[:len(mel_chunks)]
        print("->>>>>>>>>>>>>>>> Length of full_frames: {}".format(len(full_frames)))

        # self.save_detect_faces(face_det_results)

        batch_size = self.args.wav2lip_batch_size

        infer_end = int(time.time())
        print(f"->>>>>>>>>>>>>>>> inference prepare end at {infer_end}, cost time: {infer_end - infer_start}s")


        n, s_time = self.generate_frame(full_frames.copy(), mel_chunks, self.face_det_results, batch_size, self.model, self.frameNumberTotal, self.start_time_sec, myplayer)
        self.frameNumberTotal = n
        self.start_time_sec = s_time
        #time.sleep(0.1)

    def make_dir(self, dir):
        if not os.path.exists(dir): os.mkdir(dir)

    def save_detect_faces(self, face_det_results):
        return
        if not self.g_save_detect_faces:
            return
        self.g_save_detect_faces = False
        print(f'->>>>>>>>>>>>>>>> save origin detect faces and  resize to args.img_size(96*96) ')
        home_path = os.environ['HOME']

        store_dir = os.path.join(home_path, 'yulj/PycharmProjects/pythonProject/Wav2Lip-GFPGAN/outputs/detect_and_resize_faces')
        self.make_dir(store_dir)
        detect_path = os.path.join(store_dir, 'detect')
        resize_path = os.path.join(store_dir, 'resize')
        self.make_dir(detect_path)
        self.make_dir(resize_path)

        frameNumber = 0
        for  (face, coords) in face_det_results:
            cv2.imwrite(path.join(detect_path, str(frameNumber).zfill(4) + '.jpg'), face)
            resize_face = cv2.resize(face, (self.args.img_size, self.args.img_size))
            cv2.imwrite(path.join(resize_path, str(frameNumber).zfill(4) + '.jpg'), resize_face)
            frameNumber += 1
            if (frameNumber > 8): break

    def save_masked_faces(self, face_masked_results):
        return
        if not self.g_save_masked_faces:
            return
        self.g_save_masked_faces = False
        print(f'->>>>>>>>>>>>>>>> save masked detect faces after resize to args.img_size(96*96), to pred ')

        home_path = os.environ['HOME']

        store_dir = os.path.join(home_path, 'yulj/PycharmProjects/pythonProject/Wav2Lip-GFPGAN/outputs/masked_faces')
        self.make_dir(store_dir)


        frameNumber = 0
        for  mask_face in face_masked_results:
            cv2.imwrite(path.join(store_dir, str(frameNumber).zfill(4) + '.jpg'), mask_face)
            frameNumber += 1
            if (frameNumber > 8): break


    def save_pred_faces(self, face_pred_results):
        return
        if not self.g_save_pred_faces:
            return
        self.g_save_pred_faces = False
        print(f'->>>>>>>>>>>>>>>> save preded faces by masked detected faces(after resize to args.img_size(96*96) and masked) ')

        home_path = os.environ['HOME']

        store_dir = os.path.join(home_path, 'yulj/PycharmProjects/pythonProject/Wav2Lip-GFPGAN/outputs/pred_faces')
        self.make_dir(store_dir)


        frameNumber = 0
        for  pred_face in face_pred_results:
            cv2.imwrite(path.join(store_dir, str(frameNumber).zfill(4) + '.jpg'), pred_face)
            frameNumber += 1
            if (frameNumber > 8): break


    def save_pred_resize_and_connect_faces(self, face_pred_resize_results, frames, coords):
        return
        if not self.g_save_pred_resize_and_connect_faces:
            return

        self.g_save_pred_resize_and_connect_faces = False
        print(f'->>>>>>>>>>>>>>>> save resized pred-face to origin detect face size and connect to origin frame for the end result')

        home_path = os.environ['HOME']

        store_dir = os.path.join(home_path, 'yulj/PycharmProjects/pythonProject/Wav2Lip-GFPGAN/outputs/pred_resize_and_connect_faces')
        self.make_dir(store_dir)

        resize_path = os.path.join(store_dir, 'resize')
        connect_path = os.path.join(store_dir, 'connect')
        self.make_dir(connect_path)
        self.make_dir(resize_path)

        frameNumber = 0
        for p, f, c in zip(face_pred_resize_results, frames, coords):
            y1, y2, x1, x2 = c
            resizep = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            fcopy = f.copy()
            fcopy[y1:y2, x1:x2] = resizep
            cv2.imwrite(path.join(resize_path, str(frameNumber).zfill(4) + '.jpg'), resizep)
            cv2.imwrite(path.join(connect_path, str(frameNumber).zfill(4) + '.jpg'), fcopy)

            frameNumber += 1
            if (frameNumber > 8): break


    def put_frame_to_rtcq(self, buffer_q:queue.Queue, is_end:bool, myplayer:MediaPlayer, to_print:bool):

        if buffer_q.empty():
            return

        #视频帧
        den = self.denominator
        if is_end:
            den = buffer_q.qsize()

        if buffer_q.qsize() >= den:
            for i in range(0, den):
                frame = buffer_q.get()
                self.video_mq.put(frame)
                if to_print: print(f'->>>>>>>>>>>>>>>> video frame, frame:{frame}')
                if to_print: print(f'->>>>>>>>>>>>>>>> global video queue:{self.video_mq}, qsize:{self.video_mq.qsize()}')

            for i in range(0, self.numerator):
                frame1, floatArray1 = asyncio.run(self.get_audio_frames(myplayer.audio))
                if to_print: print(f'->>>>>>>>>>>>>>>> audio frame, frame:{frame1}, floatArray1:{floatArray1}')
                if to_print: print(f'->>>>>>>>>>>>>>>> global audio queue:{self.audio_mq}, qsize:{self.audio_mq.qsize()}')

                if frame1 is None:
                    return
                #frame2, floatArray2 = asyncio.run(self.get_audio_frames(myplayer.audio))
                self.audio_mq.put(floatArray1)


                #self.audio_mq.put(floatArray2)


        # 50 fps/s


        # 25/30 fps/s

        # TODO add audio frame

    def generate_frame(self, full_frames, mel_chunks, face_det_results, batch_size, model, frameNumberTotal, start_time_sec, myplayer:MediaPlayer):
        end_time_sec = int(time.time())
        frameNumber = frameNumberTotal
        gen = self.datagen(full_frames, mel_chunks, face_det_results)

        # unProcessedFramesFolderPath = "/home/ai/yulj/PycharmProjects/pythonProject/Wav2Lip-GFPGAN/after_frames"
        #
        # if not os.path.exists(unProcessedFramesFolderPath):
        #     os.makedirs(unProcessedFramesFolderPath)

        buffer_q = queue.Queue()

        to_print = True
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                        total=int(np.ceil(
                                                                            float(len(mel_chunks)) / batch_size)))):

            if to_print: print(f'->>>>>>>>>>>>>>>> batch_size {batch_size}, img_batch {len(img_batch)}, mel_batch {len(mel_batch)}, frames {len(frames)}, coords {len(coords)}')

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            if to_print: print(f'->>>>>>>>>>>>>>>> preds {len(pred)}')
            if to_print: self.save_pred_faces(pred)
            if to_print: self.save_pred_resize_and_connect_faces(pred, frames, coords)

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c

                if to_print: print(f'->>>>>>>>>>>>>>>> pred p {p.shape}, {x2 - x1}, {y2 - y1}')


                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                # out.write(f)

                buffer_q.put(f)
                self.put_frame_to_rtcq(buffer_q, False, myplayer,to_print)
                to_print = False

                # 50 fps/s

                # 25 fps/s

                # TODO add audio frame

                # self.mq.put(copy.deepcopy(f))
                # cv2.imwrite(path.join(unProcessedFramesFolderPath, str(frameNumber).zfill(4) + '.jpg'), f)
                frameNumber += 1
                if frameNumber % 125 == 0:
                    end_time_sec = int(time.time())
                    deta = end_time_sec - start_time_sec
                    self.wait_mq(300)
                    end_time_sec = int(time.time())
                    start_time_sec = end_time_sec
                    print(
                        f'->>>>>>>>>>>>>>>> {frameNumber} total frames output to mq, current video mq size: {self.video_mq.qsize()}, current audio mq size: {self.audio_mq.qsize()}, seconds: {deta}')




                if frameNumber % 500 == 0:
                    print(f'->>>>>>>>>>>>>>>> gc.collect()')
                    print(f'->>>>>>>>>>>>>>>>before: {gc.get_count()}, {gc.get_stats()}')
                    #gc.collect()
                    print(f'->>>>>>>>>>>>>>>> after: {gc.get_count()}, {gc.get_stats()}')
                    end_time_sec = int(time.time())
                    start_time_sec = end_time_sec

                if self.get_stop():
                    return frameNumber, start_time_sec

                sleep_n = 0
                sleep_chip = 0.2
                while self.get_pause():
                    time.sleep(sleep_chip)
                    sleep_n += 1

                start_time_sec += sleep_n * sleep_chip  # 把sleep的时间加上，不影响inference的时间


            del img_batch
            del mel_batch
            del pred
            del frames
            del coords

        #推送剩余的帧
        self.put_frame_to_rtcq(buffer_q, True, myplayer, to_print)

        del gen


        return frameNumber, start_time_sec
        # out.release()
        #
        # command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(self.args.audio, 'temp/result.avi', self.args.outfile)
        # subprocess.call(command, shell=platform.system() != 'Windows')
    def wait_mq(self, limit):
        sl_t = 0.2
        while self.video_mq.qsize() > limit:
            print(f'->>>>>>>>>>>>>>>> pause {sl_t}s, because video mq({self.video_mq}) data is so much({self.video_mq.qsize()}), audio mq size({self.audio_mq.qsize()}), more than {limit}, to wait webrtc to consume!!!!!!!')
            time.sleep(sl_t)

if __name__ == '__main__':
    print(f'main')
