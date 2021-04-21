import csv
import os
import sys

import cv2
import feature_extractor
import numpy
import tensorflow.compat.v1 as tf # import tensorflow as tf

# In OpenCV3.X, this is available as cv2.CAP_PROP_POS_MSEC
# In OpenCV2.X, this is available as cv2.cv.CV_CAP_PROP_POS_MSEC
CAP_PROP_POS_MSEC = cv2.CAP_PROP_POS_MSEC # 0

class YouTube8MFeatureExtractorFunc(object):

    def __init__(self, model_dir):
        self.model_dir = model_dir # 'Directory to store model files. It defaults to ~/yt8m'
        self.frames_per_second = 1  # 'This many frames per second will be processed'
        self.skip_frame_level_features = False  # 'If set, frame-level features will not be written: only
                                                # 'video-level features will be written with feature names mean_*'
        self.labels_feature_key = 'labels' #  'Labels will be written to context feature with this key, as int64 list feature.'
        self.image_feature_key = 'rgb'  # 'Image features will be written to sequence feature with
                                        # 'this key, as bytes list feature, with only one entry, '
                                        # 'containing quantized feature string.'
        self.video_file_feature_key = 'id' # Input <video_file> will be written to context feature '
                                           # 'with this key, as bytes list feature, with only one '
                                           # 'entry, containing the file path of the video. This '
                                           # 'can be used for debugging but not for training or eval.'
        self.insert_zero_audio_features = True #  'If set, inserts features with name "audio" to be 128-D '
                                               # 'zero vectors. This allows you to use YouTube-8M pre-trained model.'
        self.extractor = feature_extractor.YouTube8MFeatureExtractor(self.model_dir)

    def frame_iterator(self, filename, every_ms=1000, max_num_frames=300):
        """Uses OpenCV to iterate over all frames of filename at a given frequency.

        Args:
          filename: Path to video file (e.g. mp4)
          every_ms: The duration (in milliseconds) to skip between frames.
          max_num_frames: Maximum number of frames to process, taken from the
            beginning of the video.

        Yields:
          RGB frame with shape (image height, image width, channels)
        """
        video_capture = cv2.VideoCapture()
        if not video_capture.open(filename):
            print(str(sys.stderr) + 'Error: Cannot open video file ' + str(filename))
            return
        last_ts = -99999  # The timestamp of last retrieved frame.
        num_retrieved = 0

        while num_retrieved < max_num_frames:
            # Skip frames
            while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
                if not video_capture.read()[0]:
                    return

            last_ts = video_capture.get(CAP_PROP_POS_MSEC)
            has_frames, frame = video_capture.read()
            if not has_frames:
                break
            yield frame
            num_retrieved += 1


    def _int64_list_feature(self, int64_list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))


    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def _make_bytes(self, int_array):
        if bytes == str:  # Python2
            return ''.join(map(chr, int_array))
        else:
            return bytes(int_array)


    def quantize(self, features, min_quantized_value=-2.0, max_quantized_value=2.0):
        """Quantizes float32 `features` into string."""
        assert features.dtype == 'float32'
        assert len(features.shape) == 1  # 1-D array
        features = numpy.clip(features, min_quantized_value, max_quantized_value)
        quantize_range = max_quantized_value - min_quantized_value
        features = (features - min_quantized_value) * (255.0 / quantize_range)
        features = [int(round(f)) for f in features]

        return self._make_bytes(features)

    def video_extractor(self, given_video_file, given_labels, output_tfrecords_file, apply_pca):
        # extractor = feature_extractor.YouTube8MFeatureExtractor(self.model_dir)
        writer = tf.python_io.TFRecordWriter(output_tfrecords_file)
        total_written = 0
        total_error = 0
        if len(given_video_file) != len(given_labels):
            print('number of video and label not match')
            return False
        for video_file, labels in zip(given_video_file, given_labels):
            rgb_features = []
            sum_rgb_features = None
            for rgb in self.frame_iterator(video_file, every_ms=1000.0 / self.frames_per_second):
                features = self.extractor.extract_rgb_frame_features(rgb[:, :, ::-1], apply_pca)
                if sum_rgb_features is None:
                    sum_rgb_features = features
                else:
                    sum_rgb_features += features
                rgb_features.append(self._bytes_feature(self.quantize(features)))

            if not rgb_features:
                print(str(sys.stderr) + 'Could not get features for ' + str(video_file))
                total_error += 1
                continue

            mean_rgb_features = sum_rgb_features / len(rgb_features)

            # Create SequenceExample proto and write to output.
            feature_list = {self.image_feature_key: tf.train.FeatureList(feature=rgb_features),}
            context_features = {
                self.labels_feature_key: self._int64_list_feature(sorted(map(int, labels.split(';')))),
                self.video_file_feature_key: self._bytes_feature(self._make_bytes(map(ord, video_file))),
                'mean_' + self.image_feature_key: tf.train.Feature(float_list=tf.train.FloatList(value=mean_rgb_features)),
            }

            if self.insert_zero_audio_features:
                zero_vec = [0] * 128
                feature_list['audio'] = tf.train.FeatureList(feature=[self._bytes_feature(self._make_bytes(zero_vec))] * len(rgb_features))
                context_features['mean_audio'] = tf.train.Feature(float_list=tf.train.FloatList(value=zero_vec))

            if self.skip_frame_level_features:
                example = tf.train.SequenceExample(context=tf.train.Features(feature=context_features))
            else:
                example = tf.train.SequenceExample(context=tf.train.Features(feature=context_features),
                    feature_lists=tf.train.FeatureLists(feature_list=feature_list))
            writer.write(example.SerializeToString())
            total_written += 1
        writer.close()
        print('Successfully encoded %i out of %i videos' % (total_written, total_written + total_error))
        return True