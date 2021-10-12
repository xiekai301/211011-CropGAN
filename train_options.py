import argparse


class TrainOptions():

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    def initialize(self, parser):
        parser = argparse.ArgumentParser()

        # General Arguments / Arguments for Training
        parser.add_argument("--PATH_train", help="Path to uncropped training dataset", default='dataset/train')
        parser.add_argument("--PATH_test", help="Path to uncropped testing dataset (.tfrecord)", default='dataset/test')
        parser.add_argument("--checkpoint_dir", help="Directory to save checkpoints", default='./output/checkpoints/')
        parser.add_argument("--log_dir", help="Directory to save checkpoints", default='./output/logs/')
        parser.add_argument("--BUFFER_SIZE", help="Buffer for Shuffle", default=500)
        parser.add_argument("--BATCH_SIZE", help="Batch size", default=1)
        parser.add_argument("--IMG_WIDTH", help="Image Width", default=256)
        parser.add_argument("--IMG_HEIGHT", help="Image Height", default=256)
        parser.add_argument("--IMG_DEPTH", help="Image Depth", default=32)
        parser.add_argument("--IMG_STRIDE", help="Image Depth", default=16)
        parser.add_argument("--OUTPUT_CHANNELS", help="Image Channels (1=Grayscale)", default=1)
        parser.add_argument("--EPOCHS", help="Number of Epochs to train", default=200)
        parser.add_argument("--STEPS", help="Number of training steps per epoch", default=10)
        parser.add_argument("--GPU", help="GPU Number to use.  Currently only supports using single GPU", default=1)
        parser.add_argument("--Training", help="Are you training (True) or Predicting (False)", default=True, type=bool)

        # Arguments for Continuing Training from a Checkpoint
        parser.add_argument("--continue_training", help="Are you continuing training from a previous checkpoint?",
                            default=True, type=bool)
        parser.add_argument("--checkpoint_restore_dir", help="Path to directory of checkpoint to restore",
                            default='output/checkpoints')

        # Arguments for Inference
        parser.add_argument("--cropped_savepath", help="Path where you want to save cropped images",
                            default='output/cropped_savepath')
        parser.add_argument("--prediction_savepath", help="Path where you want to save synthetically uncropped images",
                            default='output/prediction_savepath')
        parser.add_argument("--uncropped_savepath", help="Path where you want to save  uncropped images",
                            default='output/uncropped_savepath')
        parser.add_argument("--mask_savepath", help="Path where you want to save masks used to generate cropped images",
                            default='output/mask_savepath')
        return parser

    def parse(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        else:
            parser = None
        return parser.parse_args([])
