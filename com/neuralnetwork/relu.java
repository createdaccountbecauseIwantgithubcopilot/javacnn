package com.neuralnetwork;
class relu extends activation{
    float activate(float input){
        if (input > 0){
            return input;
        }else{
            return 0;
        }
    }
    float differentiate(float input){
        if (input > 0) return 1;
        else return 0;
    }
}