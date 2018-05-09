import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Input,Dense,Dropout,Flatten,Activation
from keras.optimizers import SGD,Adam

def build_model(feat_type,num_layer,num_hid,act_func,optimizer,batch_size,epoch,nb_class):
    model = Sequential()
    if feat_type == 'lmfcc':
        input_dim = 13
    elif feat_type == 'dlmfcc':
        input_dim = 91
    elif feat_type == 'mspec':
        input_dim = 40
    elif feat_type == 'dmspec':
        input_dim = 280

    model.add(Dense(num_hid,input_shape=(input_dim,),dtype="float32"))
    for l in range(num_layer-1):
        model.add(Dense(num_hid,activation=act_func))

    model.add(Dense(nb_class))
    model.add(Activation('softmax',name='posterior'))
    if optimizer == 'adam':
        opt = Adam()
    else:
        opt = SGD()
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    model.summary()
    return model
