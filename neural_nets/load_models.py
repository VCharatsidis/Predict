import copy
import torch
from test_nn import get_means_and_stds, standardize_instance
import os


#####################  Neural nets   #####################################

def load_models(onehot_encoded):
    filepath = 'models\\'
    script_directory = os.path.split(os.path.abspath(__file__))[0]

    grubby_star = os.path.join(script_directory, filepath + 'grubbyStar/grubbyStar.model')
    grubby_star2 = os.path.join(script_directory, filepath + 'grubbyStar2/grubbyStar2.model')
    grubby_star_3L3W = os.path.join(script_directory, filepath + 'grubbyStar3L-3W/grubbyStar3L-3W.model')
    grubby_star_4L3W = os.path.join(script_directory, filepath + 'grubbyStar4L-3W/grubbyStar4L-3W.model')
    grubby_star_4L4W = os.path.join(script_directory, filepath + 'grubbyStar4L4W/grubbyStar4L4W.model')
    grubby_star_test = os.path.join(script_directory, filepath + 'grubbyStarTest.model')
    grubby_star_cross_entropy = os.path.join(script_directory, filepath + 'cross_entropy/grubbyStarCrossEntropy.model')
    grubby_ce2 = os.path.join(script_directory, filepath + 'cross_entropy/grubbyStarCE2.model')
    grubby_ce3 = os.path.join(script_directory, filepath + 'cross_entropy/grubbyStarCE3.model')
    grubby_ce4 = os.path.join(script_directory, filepath + 'cross_entropy/grubbyStarCE4.model')


    onehot_neural = copy.deepcopy(onehot_encoded)

    mean1, std1, mean2, std2 = get_means_and_stds()
    x = standardize_instance(onehot_neural, mean1, std1, mean2, std2)

    # input->2, 2->2, 2->1
    model = torch.load(grubby_star)
    model.eval()
    pred = model.forward(x)
    neural_pred = pred.detach().numpy()

    # input->3, 3->3, 3->1
    model2 = torch.load(grubby_star2)
    model2.eval()
    pred2 = model2.forward(x)
    neural_pred2 = pred2.detach().numpy()

    # input->3, 3->3, 3->3, 3->1
    model3L3W = torch.load(grubby_star_3L3W)
    model3L3W.eval()
    pred3L3W = model3L3W.forward(x)
    neural_pred3L3W = pred3L3W.detach().numpy()

    # input->3, 3->3, 3->3, 3->3, 3->1
    model4L3W = torch.load(grubby_star_4L3W)
    model4L3W.eval()
    pred4L3W = model4L3W.forward(x)
    neural_pred4L3W = pred4L3W.detach().numpy()

    # input->4, 4->4, 4->1
    model4L4W = torch.load(grubby_star_4L4W)
    model4L4W.eval()
    pred4L4W = model4L4W.forward(x)
    neural_pred4L4W = pred4L4W.detach().numpy()


    modelTest = torch.load(grubby_star_test)
    modelTest.eval()
    predTest = modelTest.forward(x)
    neural_predTest = predTest.detach().numpy()


    modelCross = torch.load(grubby_star_cross_entropy)
    modelCross.eval()
    predCross = modelCross.forward(x)
    neural_predCross = predCross.detach().numpy()

    modelCE2 = torch.load(grubby_ce2)
    modelCE2.eval()
    predCross2 = modelCE2.forward(x)
    neural_predCross2 = predCross2.detach().numpy()

    modelCE3 = torch.load(grubby_ce3)
    modelCE3.eval()
    predCross3 = modelCE3.forward(x)
    neural_predCross3 = predCross3.detach().numpy()

    modelCE4 = torch.load(grubby_ce4)
    modelCE4.eval()
    predCross4 = modelCE4.forward(x)
    neural_predCross4 = predCross4.detach().numpy()


    return neural_pred, neural_pred2, neural_pred3L3W, neural_pred4L3W, \
           neural_pred4L4W, neural_predTest, neural_predCross, neural_predCross2, neural_predCross3, neural_predCross4
