import numpy as np

from operator import truediv
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,cohen_kappa_score


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports (X_test,y_test,name,model):
    #start = time.time()
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    #end = time.time()
    #print(end - start)
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
                        'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                        'Vinyard_untrained','Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                        'Self-Blocking Bricks','Shadows']

    elif name=='HU13':
        target_names = ['Grass_healthy','Grass_stressed','Grass_synthetic','Tree','Soil','Water','Residential',
                        'Commercial','Road','Highway','Railway','Parking_lot1','Parking_lot2','Tennis_court','Running_track']

    elif name=="KSC":
        target_names = ['Scrub','Willow swamp','Cabbage palm hammock','Cabbage palm/oak hammock','Slash pine','Oak/broadleaf hammock',
                        'Hardwood swamp','Graminoid marsh','Spartine marsh','Cattail marsh','Salt marsh','Mud flats','Water']


    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)

    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)

    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)

    each_acc, aa = AA_andEachClassAccuracy(confusion)

    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)

    score = model.evaluate(X_test, y_test, batch_size=32)

    Test_Loss =  score[0]
    Test_accuracy = score[1]*100

    return classification, confusion, Test_Loss, Test_accuracy, oa*100, each_acc*100, aa*100, kappa*100