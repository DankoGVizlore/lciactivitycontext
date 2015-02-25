import numpy as np
from activity_server.models import DataRecord, AcceleratorRecord, GyroscopeRecord
from activity_server.utilities.statistics import get_features, get_features_acceleration
from activity_server.utilities.statistics import get_enhanced_features, get_enhanced_features_acceleration
from sklearn.externals import joblib
from django.core.exceptions import ObjectDoesNotExist
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, medfilt


svc_acc_gyo = joblib.load('activity_server/classifier/acc_gyo/classifier_svc.pkl')
svc_acc = joblib.load('activity_server/classifier/acc/classifier_svc.pkl')
tree_acc_gyo = joblib.load('activity_server/classifier/acc_gyo/classifier_tree.pkl')
tree_acc = joblib.load('activity_server/classifier/acc/classifier_tree.pkl')

svc_acc_gyo_ech = joblib.load('activity_server/classifier/acc_gyo_ech/classifier_svc.pkl')
svc_acc_ech = joblib.load('activity_server/classifier/acc_ech/classifier_svc.pkl')
tree_acc_gyo_ech = joblib.load('activity_server/classifier/acc_gyo_ech/classifier_tree.pkl')
tree_acc_ech = joblib.load('activity_server/classifier/acc_ech/classifier_tree.pkl')


def recognize_last_activity(uuid, algorithm, feature_set):

    record = DataRecord.objects.filter(user_id=uuid).latest('record_date')
    acceleration_data = AcceleratorRecord.objects.filter(data_record=record.id).order_by("time_stamp")

    try:
        gyroscope_data = GyroscopeRecord.objects.filter(data_record=record.id).order_by("time_stamp")
        t, x_acc, y_acc, z_acc, x_gyo, y_gyo, z_gyo = process_data(acceleration_data, gyroscope_data)

        if feature_set == 'standard':
            data = get_features(x_acc, y_acc, z_acc, x_gyo, y_gyo, z_gyo)
            if algorithm == 'svc':
                return {"vector": svc_acc_gyo.predict_proba(data)[0], "time": record.record_date}
            elif algorithm == 'dt':
                return {"vector": tree_acc_gyo.predict_proba(data)[0], "time": record.record_date}
            else:
                raise Exception('Invalid algorithm')
        elif feature_set == 'enhanced':
            data = get_enhanced_features(x_acc, y_acc, z_acc, x_gyo, y_gyo, z_gyo)
            if algorithm == 'svc':
                return {"vector": svc_acc_gyo_ech.predict_proba(data)[0], "time": record.record_date}
            elif algorithm == 'dt':
                return {"vector": tree_acc_gyo_ech.predict_proba(data)[0], "time": record.record_date}
            else:
                raise Exception('Invalid algorithm')
        else:
            raise Exception('Invalid feature set')

    except ObjectDoesNotExist:
        x, y, z, t = process_acceleration_data(acceleration_data)

        if feature_set == 'standard':
            data = get_features_acceleration(x, y, z)
            if algorithm == 'svc':
                return {"vector": svc_acc.predict_proba(data)[0], "time": record.record_date}
            elif algorithm == 'dt':
                return {"vector": tree_acc.predict_proba(data)[0], "time": record.record_date}
            else:
                raise Exception('Invalid algorithm')
        elif feature_set == 'enhanced':
            data = get_enhanced_features_acceleration(x, y, z)
            if algorithm == 'svc':
                return {"vector": svc_acc_ech.predict_proba(data)[0], "time": record.record_date}
            elif algorithm == 'dt':
                return {"vector": tree_acc_ech.predict_proba(data)[0], "time": record.record_date}
            else:
                raise Exception('Invalid algorithm')
        else:
            raise Exception('Invalid feature set')


def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(x, low_cut, high_cut, fs, order=5):
    b, a = butter_bandpass(low_cut, high_cut, fs, order=order)
    y = lfilter(b, a, x)
    return y


def filter_acceleration(x, y, z):
    x = medfilt(np.array(x))
    y = medfilt(np.array(y))
    z = medfilt(np.array(z))

    x = butter_bandpass_filter(x, 0, 20, 50, order=3)
    y = butter_bandpass_filter(y, 0, 20, 50, order=3)
    z = butter_bandpass_filter(z, 0, 20, 50, order=3)

    return x, y, z


def filter_gyroscope(x, y, z):
    x = medfilt(np.array(x))
    y = medfilt(np.array(y))
    z = medfilt(np.array(z))

    x = butter_bandpass_filter(x, 0.3, 20, 50, order=3)
    y = butter_bandpass_filter(y, 0.3, 20, 50, order=3)
    z = butter_bandpass_filter(z, 0.3, 20, 50, order=3)

    return x, y, z


def resample_acceleration_data(x, y, z, t):

    t_begin = t[0]
    t_end = t[-1]

    f_x_acc = interp1d(t, x)
    f_y_acc = interp1d(t, y)
    f_z_acc = interp1d(t, z)

    size = (t_end - t_begin)//20

    t = np.linspace(t_begin, t_begin + size * 20, size)

    x = f_x_acc(t)
    y = f_y_acc(t)
    z = f_z_acc(t)

    return t, x, y, z


def process_acceleration_data(sensor_data):
    x = []
    y = []
    z = []
    t = []

    for i in xrange(len(sensor_data)):
        x.append(sensor_data[i].x)
        y.append(sensor_data[i].y)
        z.append(sensor_data[i].z)
        t.append(sensor_data[i].t)

    x, y, z, t = resample_acceleration_data(x, y, z, t)
    x, y, z = filter_acceleration(x, y, z)

    return x, y, z, t


def resample_data(x_acc, y_acc, z_acc, t_acc, x_gyo, y_gyo, z_gyo, t_gyo):

    t_begin = max(t_acc[0], t_gyo[0])
    t_end = min(t_acc[-1], t_gyo[-1])

    f_x_acc = interp1d(t_acc, x_acc)
    f_y_acc = interp1d(t_acc, y_acc)
    f_z_acc = interp1d(t_acc, z_acc)

    f_x_gyo = interp1d(t_gyo, x_gyo)
    f_y_gyo = interp1d(t_gyo, y_gyo)
    f_z_gyo = interp1d(t_gyo, z_gyo)

    size = (t_end - t_begin)//20

    t = np.linspace(t_begin, t_begin + size * 20, size)

    x_acc = f_x_acc(t)
    y_acc = f_y_acc(t)
    z_acc = f_z_acc(t)

    x_gyo = f_x_gyo(t)
    y_gyo = f_y_gyo(t)
    z_gyo = f_z_gyo(t)

    return t, x_acc, y_acc, z_acc, x_gyo, y_gyo, z_gyo


def process_data(acceleration_data, gyroscope_data):
    x_acc = []
    y_acc = []
    z_acc = []
    t_acc = []

    x_gyo = []
    y_gyo = []
    z_gyo = []
    t_gyo = []

    for i in xrange(len(acceleration_data)):
        x_acc.append(acceleration_data[i].x)
        y_acc.append(acceleration_data[i].y)
        z_acc.append(acceleration_data[i].z)
        t_acc.append(acceleration_data[i].time_stamp)

    for i in xrange(len(gyroscope_data)):
        x_gyo.append(gyroscope_data[i].x)
        y_gyo.append(gyroscope_data[i].y)
        z_gyo.append(gyroscope_data[i].z)
        t_gyo.append(gyroscope_data[i].time_stamp)

    t, x_acc, y_acc, z_acc, x_gyo, y_gyo, z_gyo = resample_data(x_acc, y_acc, z_acc, t_acc, x_gyo, y_gyo, z_gyo, t_gyo)
    x_acc, y_acc, z_acc = filter_acceleration(x_acc, y_acc, z_acc)
    x_gyo, y_gyo, z_gyo = filter_gyroscope(x_gyo, y_gyo, z_gyo)

    return t, x_acc, y_acc, z_acc, x_gyo, y_gyo, z_gyo