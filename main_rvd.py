from ResU import *
from data_TVT import *
from keras.callbacks import *
from keras.utils import *

###############################################################################
parallel_model = unet('com.hdf5',(512,512,3))
loss_fun=com_loss()
#parallel_model = multi_gpu_model(model, gpus=2)

parallel_model.compile(optimizer = Adam(lr = 1e-4), loss = loss_fun, metrics = ['accuracy'])#'binary_crossentropy''accuracy'


checkpoint = ModelCheckpoint('com.hdf5', monitor='loss',
                                        verbose=1, save_best_only=True, mode='min')#Tumor_ResU_rvd.hdf5

#reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2,
 #                               min_lr=0.00001, min_delta=0.0001, mode='min', verbose=1)

csv_logger = CSVLogger('./com.out', append=True, separator=';')#./Tumor_result_rvd.out

earlystopping = EarlyStopping(monitor = 'loss', verbose = 1,
                                         min_delta = 0.0001, patience = 5, mode = 'min',restore_best_weights=True)

callbacks_list = [checkpoint, csv_logger, earlystopping]

train_data=trainGenerator(size=(512,512),batch=1,
                                    image_path='/home/rak038/Segmentation/U-net/data2/Tumor/Train/image_seg',
                                    mask_path='/home/rak038/Segmentation/U-net/data2/Tumor/Train/tumormasks')
val_data=valGenerator(size=(512,512),batch=1,
                                    image_path='/home/rak038/Segmentation/U-net/data2/Tumor/Val/image_seg',
                                    mask_path='/home/rak038/Segmentation/U-net/data2/Tumor/Val/tumormasks')
results = parallel_model.fit_generator(train_data,epochs=2000,
                                    steps_per_epoch = 1914, #(NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                                    validation_data=val_data,
                                    validation_steps= 459,  #(NO_OF_VAL_IMAGES//BATCH_SIZE),
                                    callbacks=callbacks_list)


test_generator = testGenerator(test_path='/home/rak038/Segmentation/U-net/data2/Tumor/Test/image_seg/0/',target_size=(512,512))
results = parallel_model.predict_generator(test_generator,429,verbose=1)
saveResult("/home/rak038/Segmentation/U-net/data2/Tumor/Test_seg_com/",results)
