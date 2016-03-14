#!/usr/bin/python
import numpy as np
from os import listdir
from os.path import isfile, join
import re
from copy import deepcopy
from itertools import chain
import cPickle as p
import random

# /afs/inf.ed.ac.uk/user/s12/s1235260/.local/lib/python2.7/site-packages/sklearn/neural_network/
# model3.pkl is the current benchmark with 0.93475877193 acc
class ExerciseDataProvider:
    """
    The following data provider is here to parse
    the tree-folder styled data set in to a dictionary
    which can be then used as a proper training set.

    # yoloswags
    """

    class_dict = {
                    'a01': 'sitting',
                    'a02': 'standing',
                    'a03': 'lying_on_back',
                    'a04': 'lying_on_right_side',
                    'a05': 'ascending_stairs',
                    'a06': 'descending_stairs',
                    'a07': 'standing_in_an_elevator_still',
                    'a08': 'moving_around_in_an_elevator',
                    'a09': 'walking_in_a_parking_lot',
                    'a10': 'walking_on_a_treadmill_with_a_speed_of_4km/h_in_flat',
                    'a11': 'deg_inclined_positions',
                    'a12': 'running_on_a_treadmill_with_a_speed_of_8km/h',
                    'a13': 'exercising_on_a_stepper',
                    'a14': 'exercising_on_a_cross_trainer',
                    'a15': 'cycling_on_an_exercise_bike_horizontal',
                    'a16': 'cycling_on_an_exercise_bike_vertical',
                    'a17': 'rowing',
                    'a18': 'jumping',
                    'a19': 'playing_basketball'
    }


    def __init__(self,
                 base_path="."):
        self.base_path = base_path
        self.data_dict, self.rand_dict, self.rand_dict1 = self.get_data_dict()
        self.x , self.t, self.xt, self.tt, self.Xv, self.Tv = self.p()
        # ((self.x , self.t),(self.xt , self.tt)) = self.prepare_train_set()

    def p(self):
        X = np.array(map(lambda x: x[0],
                     self.data_dict.values()))
        print X.shape
        X = X.reshape(-1,X.shape[-2],X.shape[-1])
        print X.shape

        T = np.array(map(lambda x: x[1],
                     self.data_dict.values())).\
                     reshape(-1)


        Xt = np.array(map(lambda x: x[0],
                     self.rand_dict.values()))

        Xt = Xt.reshape(-1,Xt.shape[-2],Xt.shape[-1])
        Tt = np.array(map(lambda x: x[1],
                     self.rand_dict.values())).\
                     reshape(-1)

        Xv = np.array(map(lambda x: x[0],
                     self.rand_dict1.values()))
        Xv = Xv.reshape(-1,Xv.shape[-2],Xv.shape[-1])
        Tv = np.array(map(lambda x: x[1],
                     self.rand_dict1.values())).\
                     reshape(-1)

        return X, T, Xt, Tt, Xv, Tv


    def get_data_dict(self):
        mypath_on = self.base_path
        folds = [f
                 for f
                 in listdir(mypath_on)
                 if re.match('a*',f).group(0)]
        files_dict = dict()
        random_dict = dict()
        random_dict1 = dict()
        for ii, fold in enumerate(folds):
            people_path = mypath_on + "/" + fold
            people =[f for f in listdir(people_path)]
            if ii == 0:
                random.shuffle(people)
                pd = people.pop(0)
                pd = [pd, people.pop(0)]
            else:
                people.remove(pd[0])
                people.remove(pd[1])
            files = list()

            fp = [join(people_path, p) for p in pd]
            print fp
            print  join(fp[0],f)
            filer = [join(fp[0],f) for f in listdir(fp[0])
                          if isfile(join(fp[0],f))]

            parsed_r = list()
            # ford
            print filer
            for f in filer:
                with open(f,'rb') as fo:
                        ford = np.array(map(lambda x:map(float, x.split(",")),
                                        fo.read().split("\n")[:-1]))
                        # print ford.shape
                        parsed_r.append(deepcopy(np.array(ford).T ))
            # print len(parsed_r)
            filer = [join(fp[1],f) for f in listdir(fp[1])
                          if isfile(join(fp[1],f))]
            parsed_r2 = list()
            # ford
            print filer
            for f in filer:
                with open(f,'rb') as fo:
                        ford = np.array(map(lambda x:map(float, x.split(",")),
                                        fo.read().split("\n")[:-1]))
                        # print ford.shape
                        parsed_r2.append(deepcopy(np.array(ford).T ))
            for p in people:
                file_path = join(people_path, p)
                files.append([join(file_path,f)
                              for f in listdir(file_path)
                              if isfile(join(file_path,f))])
            files = list(chain(*files))
            parsed_files = list()
            for f in files:
                with open(f,'rb') as fo:
                    ford = np.array(map(lambda x:map(float, x.split(",")),
                                    fo.read().split("\n")[:-1]))

                    parsed_files.append(deepcopy(np.array(ford).T ))
            assert len(parsed_files) == len(files)
            labl =  int(''.join((x for x in fold if x.isdigit())))
            # print self.class_dict[fold]
            # print np.array(parsed_files).shape
            # hkjfdhfdkjhfds
            # print (np.array(parsed_files),
            #      np.array([labl]*len(files)) )
            files_dict[fold] = (np.array(parsed_files),
                                np.array([labl]*len(files)))
            random_dict[fold] = (np.array(parsed_r),
                                 np.array([labl]*len(filer)))
            random_dict1[fold] = (np.array(parsed_r2),
                                  np.array([labl]*len(filer)))
        return files_dict, random_dict, random_dict1





    def randomize(self):
       raise BaseException("TODO")






if  __name__ == "__main__":
    # from sklearn.neural_network.multilayer_perceptron import MLPClassifier
    import sys

    obj = ExerciseDataProvider(".")
    X = obj.x
    y = obj.t
    Xt = obj.xt
    yt = obj.tt
    Xv = obj.Xv
    yv = obj.Tv

    with open('/afs/inf.ed.ac.uk/user/s12/s1235260/ACL1_trainv.pkl', 'wb') as f:
        p.dump((X,y), f)

    with open('/afs/inf.ed.ac.uk/user/s12/s1235260/ACL1_testv.pkl', 'wb') as f:
        p.dump((Xt,yt), f)

    with open('/afs/inf.ed.ac.uk/user/s12/s1235260/ACL1_validv.pkl', 'wb') as f:
        p.dump((Xv,yv), f)
