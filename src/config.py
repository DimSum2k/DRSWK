import argparse
import sys
import os
import time

def get_arguments():
    parser = argparse.ArgumentParser(description="Training and evaluation configuration options.")
    
    parser.add_argument('--path_data_raw', help='path to raw MNIST data', default="../data/mnist_raw/", type=str)
    parser.add_argument('--path_data_fashion_raw', help='path to raw fashion MNIST data', default="../data/fashion_mnist_raw/", type=str)
    parser.add_argument('--path_data', help='path to save processed data', default="../data/mnist/", type=str)

    parser.add_argument('--build_data', default=False, help='if True, build dataset from raw data files', action="store_true")
    parser.add_argument('--classes', default="0123456789", help='selected classes for classification', type=str)
    parser.add_argument('--non_uniform', default=False,
                        help='weight points by pixel intensities when building histograms when activated',
                        action='store_true')
    parser.add_argument('--fashion', default=False,
                        help='use Fashion mnist insitead of mnist when activated',
                        action='store_true')

    parser.add_argument('--session_idx', default=500, help='index of the training session to save results', type=int)
    parser.add_argument('--T_train', default=1000, help="size of the train set", type=int)
    parser.add_argument('--T_val', default=300, help="size of the validation set", type=int)
    parser.add_argument('--T_test', default=500, help="size of the test set ", type=int)
    parser.add_argument('--kernel', help="select which kernel to use", type=str)
    # for building gram matrices
    parser.add_argument('--build_gram', default=False, help="if True build the train and train/test gram matrices",
                        action='store_true')
    parser.add_argument('--n_gammas', default=14, help="number of length-scale to test on Gaussian-like kernels", type=int)
    parser.add_argument('--n_degrees', default=10, help="degrees to test for the MMD polynomial kernel",type=int)
    parser.add_argument('--n_slices', default=100, help="number of slices for SW kernels", type=int)
    parser.add_argument('--n_ts', default=100, help="number of points on the real line for RF-SW kernel", type=int)
    parser.add_argument('--deterministic', default=False, help="if true take a deterministic grid for the SW-RF kernel", action="store_true")
    parser.add_argument('--true_rf', default=False, help="if true use true RF sampling",
                        action="store_true")
    # for training
    parser.add_argument('--train', default=False, help='if True, train and fine-tuned on pre-built gram matrices', action="store_true")
    parser.add_argument('--n_param_clf', default=25, help="number of hyperparameters to try for classifiers", type=int)
    parser.add_argument('--subsample', default=1, help="Train on smaller training set", type=int)
    # for evaluation
    parser.add_argument('--evaluate', default=False, help='if True, evaluate fine-tuned classifier on test set',
                        action="store_true")
    return parser

def check_args(opt):
    """Parse the arguments and perform a few unit tests."""
    opt.path_data = opt.path_data + "classes_{}".format(opt.classes)
    opt.path_data += "_fashion" * opt.fashion + "/"
    opt.classes = [int(d) for d in opt.classes]
    opt.non_uniform=True
    if opt.true_rf:
        opt.n_ts = opt.n_slices
    if opt.train or opt.build_gram or opt.evaluate:
        assert (opt.T_train // opt.subsample) % len(opt.classes) == 0
        if opt.kernel not in ["standard",
                              "Hellinger",
                              "MMD_gauss_gauss",
                              "SW_gauss_rf",
                              "SW_1_gauss_rf"]:
            print("Unknown kernel")
            sys.exit(0)
        session_name = "../results/mnist/{}/{}_{}_{}_{}".format(opt.kernel,
                                                                 opt.session_idx,
                                                                 "".join(str(d) for d in opt.classes),
                                                                 opt.T_train,
                                                                 opt.T_val)
        session_name += "_non_uniform"*opt.non_uniform
        session_name += "_fashion" * opt.fashion + "/"
        print("Session name : {}".format(session_name))
        return opt, session_name
    else:
        return opt, ""

def check_savings(opt, session_name):
    """Handle saving system."""
    if opt.build_data:
        if not os.path.exists(opt.path_data):
            os.makedirs(opt.path_data)
        if os.path.exists(opt.path_data + 'train_data.pickle') or os.path.exists(opt.path_data + 'test_data.pickle'):
            answer = input("Data already exist, are you sure you want to overwrite? [y/n] ")
            if answer != "y":
                sys.exit(0)

    if opt.train or opt.build_gram or opt.evaluate:
        if not os.path.exists(session_name + 'logs/'):
            os.makedirs(session_name + 'logs/')
        if not os.path.exists(session_name):
            os.makedirs(session_name)
        if opt.build_gram:
            if not os.path.exists(session_name + 'gram_matrices/'):
                os.makedirs(session_name + 'gram_matrices/')
            if len(os.listdir(session_name + 'gram_matrices/')) != 0:
                answer = input('Gram matrices already exist, are you sure you want to overwrite? [y/n] ')
                if answer != 'y':
                    sys.exit(0)
        if opt.train:
            if not os.path.exists(session_name + 'gram_matrices/') or (len(os.listdir(session_name + 'gram_matrices/')) == 0):
                if not opt.build_gram:
                    print('Empty gram_matrices folder, you should run --build_gram, EXIT.')
                    sys.exit(0)
            if not os.path.exists(session_name + 'training_results/'):
                os.makedirs(session_name + 'training_results/')
            if len(os.listdir(session_name + 'training_results/')) != 0:
                answer = input('Training session already exists, are you sure you want to overwrite? [y/n] ')
                if answer != 'y':
                    sys.exit(0)
        if opt.evaluate:
            if not os.path.exists(session_name + 'training_results/') or (len(os.listdir(session_name + 'training_results/')) == 0):
                if not opt.train:
                    print('Empty training folder, you should run --train, EXIT.')
                    sys.exit(0)
            if not os.path.exists(session_name + 'evaluation/'):
                os.makedirs(session_name + 'evaluation/')

        timestr = time.strftime("%Y%m%d-%H%M%S")
        with open(session_name + 'logs/commandline_args_{}.txt'.format(timestr), 'w') as f:
            f.write("User arguments: \n\n")
            f.write('\n'.join(sys.argv[1:]))
            f.write("\n\nFull arguments: \n\n")
            for k, v in vars(opt).items():
                f.write("--{}\n".format(k))
                f.write(str(v) + "\n")
