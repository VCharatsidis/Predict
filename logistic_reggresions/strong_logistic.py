from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


def strong_logistic(games, logistic_input, xin, original_input):
    strong_logistic = 0
    strong_logistic_CV = 0

    if games <= 15:
        logistic_input15 = [a[:-1] for a in logistic_input if a[-1] <= 20]
        y_15 = [a[0] for a in original_input if int(a[-2]) <= 20]
        clf15 = LogisticRegression(solver='lbfgs', max_iter=1000).fit(logistic_input15, y_15)
        clf15CV = LogisticRegressionCV(solver='lbfgs', max_iter=1000, cv=10).fit(logistic_input15, y_15)

        y_pred_logistic15_CV = clf15CV.predict_proba([xin[:-1]])
        y_pred_logistic15 = clf15.predict_proba([xin[:-1]])

        strong_logistic = y_pred_logistic15[0][1]
        strong_logistic_CV = y_pred_logistic15_CV[0][1]

    elif games < 40:
        logistic_input35 = [a[:-1] for a in logistic_input if 16 <= a[-1] <= 60]
        y_35 = [a[0] for a in original_input if 16 <= int(a[-2]) <= 60]
        clf35 = LogisticRegression(solver='lbfgs', max_iter=1000).fit(logistic_input35, y_35)
        clf35CV = LogisticRegressionCV(solver='lbfgs', max_iter=1000, cv=10).fit(logistic_input35, y_35)

        y_pred_logistic15_CV = clf35CV.predict_proba([xin[:-1]])
        y_pred_logistic35 = clf35.predict_proba([xin[:-1]])

        strong_logistic = y_pred_logistic35[0][1]
        strong_logistic_CV = y_pred_logistic15_CV[0][1]

    else:
        logistic_inputRest = [a[:-1] for a in logistic_input if 40 <= a[-1]]
        y_Rest = [a[0] for a in original_input if 40 <= int(a[-2])]
        clfRest = LogisticRegression(solver='lbfgs', max_iter=400).fit(logistic_inputRest, y_Rest)
        clfRest_CV = LogisticRegressionCV(solver='lbfgs', max_iter=400, cv=10).fit(logistic_inputRest, y_Rest)

        y_pred_logisticRest = clfRest.predict_proba([xin[:-1]])
        y_pred_logisticRestCV = clfRest_CV.predict_proba([xin[:-1]])

        strong_logistic = y_pred_logisticRest[0][1]
        strong_logistic_CV = y_pred_logisticRestCV[0][1]

    return strong_logistic, strong_logistic_CV


