
import os, sys, re, glob, argparse
import torch
import models, loss
import forward_models.forward_models as forward_models


# ===========================
# Boilerplate Code
# ===========================

def get_profile():

    # Model
    parser = argparse.ArgumentParser(description='Profile Selector')
    parser.add_argument('profile', default='Noise_DnCNN')
    args = parser.parse_args(sys.argv[1:2])

    profile = getattr(sys.modules[__name__], args.profile)
    return profile()

class DefaultProfile:

    description = 'default'

    # NN Model
    model = models.DnCNN
    model_params = {'residual_learn':True}

    # Forward Model
    forward_model = forward_models.GaussianNoise
    forward_params = {'sigma':25}

    # Loss
    loss = loss.sum_squared_error

    # Data
    patch_size = 40
    stride = 10
    scales = [1, 0.9, 0.8, 0.7]
    aug_times = 1

    # Training
    train_data = 'data/Train/Train400'
    batch_size = 128
    lr = 1e-3
    epoch = 180

    # Test
    test_dir = 'data/Test'
    test_sets = ['Set12']
    result_dir = 'results'
    save_result = 0
    model_name = 'model.pth'

    def __init__(self):
        
        # One-off Modifications
        self.parse_args()

    def parse_args(self):

        parser = argparse.ArgumentParser(description='PyTorch DnCNN')
        parser.add_argument('--description', default=self.description, help='description of modifications')
        # Model Params
        parser.add_argument('--model_params', default=list(), nargs='*', help='use residual learning, 1 or 0')
        parser.add_argument('--forward_params', default=list(), nargs='*', help='parameters for degradation model')
         # Data
        parser.add_argument('--patch_size', default=40, help='training image patch size')
        parser.add_argument('--scales',default=self.scales, help='image rescaling')
        parser.add_argument('--aug_times',default=1,help='number of times to augment image patch (rotation,etc.)')
        # Training
        parser.add_argument('--train_data', default=self.train_data, type=str, help='path of train data')
        parser.add_argument('--batch_size', default=self.batch_size, type=int, help='batch size')
        parser.add_argument('--epoch', default=self.epoch, type=int, help='number of train epoches')
        parser.add_argument('--lr', default=self.lr, type=float, help='initial learning rate for Adam')
        # Test
        parser.add_argument('--test_dir', default=self.test_dir, type=str, help='directory of test dataset')
        parser.add_argument('--test_sets', default=self.test_sets, help='directory of test dataset')
        parser.add_argument('--result_dir', default=self.result_dir, type=str, help='directory of test dataset')
        parser.add_argument('--save_result', default=self.save_result, type=int, help='save the reconstructed image, 1 or 0')
        parser.add_argument('--model_name', default=self.model_name, type=str, help='the model name (e.g. model_001.pth)')
        args = parser.parse_args(sys.argv[2:])
        
        if args.model_params:
            self.model_params = dict(zip(args.model_params[::2],args.model_params[1::2]))
            self.model_params = self.model.validate_args(**self.model_params)
        if args.forward_params:
            self.forward_params = dict(zip(args.forward_params[::2],args.forward_params[1::2]))
            self.forward_params = self.forward_model.validate_args(**self.forward_params)

        args = vars(args)
        del args['model_params']
        del args['forward_params']

        for key in args:
            setattr(self, key, args[key])

    def todict(self):

        keys = ['description','model_params','forward_params','patch_size','stride','scales','aug_times','train_data','batch_size','lr']
        d = {key:getattr(self, key) for key in keys}
        d.update({key:str(getattr(self, key)) for key in ['model','forward_model','loss']})
        return d


    def tocsv(self,model_dir):
        import csv
        
        def writedict(w,d):
            for key, val in d.items():
                if isinstance(val,dict):
                    w.writerow([key,'begin'])
                    writedict(w,val)
                    w.writerow([key,'end'])
                else:
                    w.writerow([key, val])

        with open(os.path.join(model_dir,"profile.csv"), "w") as f:
            w = csv.writer(f)
            writedict(w,self.todict())
        
    def tojson(self,model_dir):
         
        import json
        
        json = json.dumps(self.todict())
        with open(os.path.join(model_dir,"profile.json"),"w") as f:
            f.write(json)

    def compare(self,model_dir):

        import json 

        with open(os.path.join(model_dir,"profile.json")) as f:
            json = json.load(f)
        
        return json == self.todict()


    def tostring(self):
        return type(self).__name__+'-' + self.description

    def get_forward_model(self):
        forward_model = self.forward_model(**self.forward_params)
        return forward_model

    def get_test_model(self):
        model_path = os.path.join('models', self.tostring(), self.model_name)
        assert(os.path.exists(model_path))
        model = torch.load(model_path)
        return model

    def get_result_dir(self):
        if not os.path.exists(self.result_dir): os.mkdir(self.result_dir)
        
        result_dir = os.path.join(self.result_dir, self.tostring() + '-' + os.path.splitext(self.model_name)[0])
        if not os.path.exists(result_dir): os.mkdir(result_dir)
        
        return result_dir

    def get_model(self):

        model_dir = self.get_model_dir()

        initial_epoch = findLastCheckpoint(save_dir=model_dir)  # load the last model in matconvnet style
        if initial_epoch > 0:
            print('resuming by loading epoch %03d' % initial_epoch)
            model = torch.load(os.path.join(model_dir, 'model_%03d.pth' % initial_epoch))
        else:
            model = self.model(**self.model_params)

        return model,model_dir,initial_epoch

    def get_model_dir(self):

        model_dir = os.path.join('models', self.tostring())
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
            self.tocsv(model_dir)
            self.tojson(model_dir)
        else:
            assert self.compare(model_dir), 'Profile does not match existing model. Please modify description'

        return model_dir


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

# ===========================
# User-defined Profiles
# ===========================

class Isoplanatic_DnCNN(DefaultProfile):

    forward_model = forward_models.Isoplanatic
    forward_params = {'dr0':10, 'sig':0, 'speckle_flg':False, 'padsize':10}

class Noise_DnCNN(DefaultProfile):

    pass