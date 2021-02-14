from optparse import OptionParser

# This config file is for both the classifier and the pietro_style_transfer file
parser = OptionParser()


parser.add_option("--mode", dest="mode", default="train",
                  help="set to train or eval")
parser.add_option("--cuda", dest="cuda", type='int', default=1,
                  help="set it to 1 for running on GPU, 0 for CPU")
parser.add_option("--gpu-used", dest="gpu_used", type='str', default='3',
                  help="set gpu number to use for training (default 0)")

# Style data augmentation
parser.add_option('--style-augm', dest='style_augm', default=1, type='int',
                  help='train with or without style augmented dataset (default 1)')
parser.add_option('--style-ref_src', dest='style_ref_src', default='init', type='str',
                  help='where to get styles to transfer to (init or style_folder)')

# Train options
parser.add_option("--epochs", dest='epochs', type='int', default=500,
                  help="number of training epochs, default is 100")
parser.add_option("--batch-size", dest='batch_size', type='int', default=4,
                  help="batch size for training, default is 16. Must be <= num_styles for combined model")

parser.add_option("--mix-styles", dest='mix_styles', type='int', default=1,
                  help="whether or not to mix 3 styles 1")
parser.add_option("--style-entropy", dest='style_entropy', type='int', default=0,
                  help="whether or not to initialize as normal distribution instead of all ones")

# Weights and Hyperparameters
parser.add_option("--stack-size", dest='stack_size', type='int', default=8,
                  help="number of images sampled from one patient")
parser.add_option("--num-styles", dest='num_styles', type='int', default=27,
                  help="number of style images used during training, default is 12")
parser.add_option("--content-weight", dest='cont_weight', type='float', default=1,
                  help="weight for content loss, default is 1e5")
parser.add_option("--style-weight", dest='style_weight', type='float', default=1e4,
                  help="weight for style loss, default is 1e10")
parser.add_option("--classifier-weight", dest='classifier_weight', type='float', default=1,
                  help="weight for classifier loss, default is 1")
parser.add_option("--seed", dest='seed', type='int', default=42,
                  help="random seed for training, default is 42")
parser.add_option("--lr", dest='lr', type='float', default=1e-4,
                  help="learning rate, default is 1e-3")

# Optimizer options
parser.add_option('--single-optimizer', dest='single_optimizer', default='Adam', type='str',
                  help='optimizer to use for combined model (Adam, SGD)')
parser.add_option('--two-optimizers', dest='two_optimizers', default=1, type='int',
                  help='use dual optimizers (default: 1)')

###############
# AMR OPTIONS #
###############
parser.add_option('--lm', '--label-mode', dest='label_mode', default='cons', type='str',
                  help='Label Mode, either consensus (cons) or sampling (samp) (default cons)')
parser.add_option("--native_labels", dest='native_labels', default=1, type='int',
                  help="Weight of native labels (i.e. lab of origin labels, default = 1)")
parser.add_option('--num_classes', dest='num_classes', default=2, type='int',
                  help='number of classes to predict, 2 or 3 (default: 3)')

####################
# MorphSet OPTIONS #
####################
parser.add_option('--hea', '--heads', dest='heads', default=1, type='int',
                  help='The number of attention heads in the set operations.')
parser.add_option('--enc', '--enc', dest='encoder', default='efficientnet-b3', type='str',
                  help='The feature encoder prior to our setformer operation (default: efficientnet-b7)')
parser.add_option('--prsc', '--prsc', dest='preset_channels', default=256, type='int',
                  help='The number of channels prior to the set operations, must have a perfect root.')
parser.add_option('--sp', '--sp', dest='set_points', default=8, type='int',
                  help='The number of unique points (or seeds), morphological appearances.')
parser.add_option('--posc', '--posc', dest='postset_channels', default=512, type='int',
                  help='The number of channels after the set operations.')
parser.add_option('--vi', '--vi', dest='val_iters', default=10, type='int',
                  help='The number of times we repeat val to produce our final labels.')


# Dataset
parser.add_option("--dataset", dest='dataset',
                  default="/home/cougarnet.uh.edu/pcicales/Documents/data/ABMR_dataset/AMR_raw_gloms/",
                  help="path to dataset")
parser.add_option("--dtf", dest='data_folds',
                  default="/home/cougarnet.uh.edu/pcicales/Documents/data/ABMR_dataset/folds/",
                  help="path to npz fold files")
parser.add_option('--loo', '--loo', dest='test_fold_val', default=2, type='int',
                  help='Testing Fold (default 0)')

# Directories
parser.add_option("--save-dir", dest='save_dir', default='./save',
                  help='saving directory of training information (default: ./save/combined)')
parser.add_option('--classifier-load_model_path', dest='classifier_load_model_path',
                  default='/home/cougarnet.uh.edu/sdpatiba/Desktop/AMR_style_transfer_project/'
                          'save/classifier/20201212_182753/models/5143.ckpt',
                  help='path to load a .ckpt model')

# Log and Validation Intervals/Frequencies
parser.add_option("--log-interval", dest='log_interval', type='int', default=5,
                  help="number of images after which all losses are logged, default is 40")
parser.add_option('--vf', '--val_freq', dest='val_freq', default=10, type='int',
                  help='run validation for each <val_freq> iterations (default: 200)')

# Classifier Options
parser.add_option('--classifier-model', dest='classifier_model', default='morphset',
                  help='vgg, inception, resnet, densenet, morphset (default: densenet)')
parser.add_option('--j', '--num-workers', dest='num_workers', default=16, type='int',
                  help='number of data loading workers (default: 16)')

# Image size
parser.add_option('--ih', '--img_h', dest='img_h', default=256, type='int',
                  help='input image height (default: 256)')
parser.add_option('--iw', '--img_w', dest='img_w', default=256, type='int',
                  help='input image width (default: 256)')
parser.add_option('--ic', '--img_c', dest='img_c', default=3, type='int',
                  help='number of input channels (default: 3)')

# Stylize Images
parser.add_option("--content-image", dest='stylize_content_image',
                  default="/home/cougarnet.uh.edu/srizvi7/Desktop/AMR_project_style_transfer/"
                          "images/AMR_samples/ABMR-18c41473x400PAS-003.jpg",
                  help="path to content image you want to stylize")
parser.add_option("--output-img-path", dest='stylize_output_img_path', default="./images/combined_network/",
                  help="path for saving the output image")
parser.add_option("--stylize-model", dest='stylize_model',
                  default="./ckpt_epoch_15.pth",
                  help="saved model to be used for stylizing the image.")
parser.add_option("--stylize-no", dest='stylize_eval_no', default=0, type='int',
                  help="style index used for stylizing the image during evaluation time.")


options, _ = parser.parse_args()
