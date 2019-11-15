import copy
import torch
import os
import numpy as np
from torch.autograd import Variable
from neural_nets.input_to_onehot import input_to_onehot
from input_cross_entropy import cross_entropy_input_to_onehot


#####################  Neural nets   #####################################


def get_means_and_stds(X_train):
    mean1 = np.mean(X_train[:, -1], axis=0)
    std1 = np.std(X_train[:, -1], axis=0)

    mean2 = np.mean(X_train[:, -2], axis=0)
    std2 = np.std(X_train[:, -2], axis=0)

    return mean1, std1, mean2, std2


def standardize_instance(onehot_neural, mean1, std1, mean2, std2):
    onehot_neural = onehot_neural.astype(float)

    onehot_neural[-1] -= mean1
    k = onehot_neural[-1] / std1
    onehot_neural[-1] = k

    onehot_neural[-2] -= mean2
    z = onehot_neural[-2] / std2
    onehot_neural[-2] = z

    xin = Variable(torch.FloatTensor([onehot_neural]))

    return xin


def load_models(onehot_encoded):
    filepath = 'models\\'
    script_directory = os.path.split(os.path.abspath(__file__))[0]

    grubby_star = os.path.join(script_directory, filepath + 'grubbyStar/grubbyStar.model')
    grubby_star2 = os.path.join(script_directory, filepath + 'grubbyStar2/grubbyStar2.model')
    grubby_star_3L3W = os.path.join(script_directory, filepath + 'grubbyStar3L-3W/grubbyStar3L-3W.model')
    grubby_star_4L3W = os.path.join(script_directory, filepath + 'grubbyStar4L-3W/grubbyStar4L-3W.model')
    grubby_star_4L4W = os.path.join(script_directory, filepath + 'grubbyStar4L4W/grubbyStar4L4W.model')

    grubby_star_cross_entropy = os.path.join(script_directory, filepath + 'cross_entropy/grubbyStarCrossEntropy.model')
    grubby_ce2 = os.path.join(script_directory, filepath + 'cross_entropy/grubbyStarCE2.model')
    grubby_ce3 = os.path.join(script_directory, filepath + 'cross_entropy/grubbyStarCE3.model')
    grubby_ce4 = os.path.join(script_directory, filepath + 'cross_entropy/grubbyStarCE4.model')

    metamodel = os.path.join(script_directory, "metamodel/" + 'grubbyStarMeta.model')
    sigmamodel = os.path.join(script_directory, "sigma_metamodel/" + 'grubbyStarSigma.model')
    enhanced = os.path.join(script_directory, filepath + "enhanced_features/" + 'enhancedStar.model')

    onehot_neural = copy.deepcopy(onehot_encoded)
    _, _, X_train = input_to_onehot('automagic')
    mean1, std1, mean2, std2 = get_means_and_stds(X_train)
    x = standardize_instance(onehot_neural, mean1, std1, mean2, std2)

    onehot_neural_cross = copy.deepcopy(onehot_encoded)
    _, _, X_train_cross = cross_entropy_input_to_onehot()
    mean1_cross, std1_cross, mean2_cross, std2_cross = get_means_and_stds(X_train_cross)
    x_cross = standardize_instance(onehot_neural_cross, mean1_cross, std1_cross, mean2_cross, std2_cross)

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


    ######### CROSS ##################


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

    # print("prediction", pred)
    # print("x.shape", x.shape)
    # print("pred.shape", pred.shape)
    third_tensor = torch.cat((x, pred2), 1)
    third_tensor = torch.cat((third_tensor, pred4L3W), 1)
    # third_tensor = torch.cat((third_tensor, pred3L3W), 1)
    # third_tensor = torch.cat((third_tensor, pred4L3W), 1)
    # third_tensor = torch.cat((third_tensor, pred4L4W), 1)

    third_tensor = torch.cat((third_tensor, predCross), 1)
    # third_tensor = torch.cat((third_tensor, predCross2), 1)
    third_tensor = torch.cat((third_tensor, predCross3), 1)
    # third_tensor = torch.cat((third_tensor, predCross4), 1)

    metaStar = torch.load(metamodel)
    metaStar.eval()
    pred_meta = metaStar.forward(third_tensor)
    coeffs = pred_meta.detach().numpy()

    third_tensor = torch.narrow(third_tensor, 1, 25, 4)

    result = torch.mul(third_tensor, pred_meta)
    result = torch.sum(result, dim=1)
    neural_meta = result.detach().numpy()


    sigma_tensor = torch.cat((x, pred), 1)
    sigma_tensor = torch.cat((sigma_tensor, pred2), 1)
    sigma_tensor = torch.cat((sigma_tensor, pred3L3W), 1)
    sigma_tensor = torch.cat((sigma_tensor, pred4L3W), 1)
    sigma_tensor = torch.cat((sigma_tensor, pred4L4W), 1)

    sigma_tensor = torch.cat((sigma_tensor, predCross), 1)
    sigma_tensor = torch.cat((sigma_tensor, predCross2), 1)
    sigma_tensor = torch.cat((sigma_tensor, predCross3), 1)
    sigma_tensor = torch.cat((sigma_tensor, predCross4), 1)

    sigmaStar = torch.load(sigmamodel)
    sigmaStar.eval()
    pred_sigma = sigmaStar.forward(sigma_tensor)

    train = torch.narrow(sigma_tensor, 1, 25, 9)

    mean = torch.mean(train, dim=1)
    s = torch.std(train, dim=1)

    result_sigma = mean + torch.mul(torch.transpose(pred_sigma, 0, 1), s)

    result_sigma = result_sigma.detach().numpy()


    enhancedStar = torch.load(enhanced)
    enhancedStar.eval()
    pred_enha = enhancedStar.forward(sigma_tensor)
    pred_enha = pred_enha.detach().numpy()

    return neural_pred, neural_pred2, neural_pred3L3W, neural_pred4L3W, \
           neural_pred4L4W, neural_predCross, neural_predCross2, neural_predCross3, neural_predCross4, neural_meta, coeffs[0], result_sigma[0], pred_enha
