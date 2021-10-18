root = {

    "general" : {

        "display_viewer" : False,
        #The visible GPUS will be restricted to the numbers listed here. The pytorch (cuda:0) numeration will start at 0
        #This is a trick to get everything onto the wanted gpus because just setting cuda:4 in the function calls will
        #not work for mmdetection. There will still be things on gpu cuda:0.
        "cuda_visible_devices" : "3",
        "save_track_results" : True

    },

    "data" : {
        # To increase the speed while developing an specific interval of all frames can be set.
        "selection_interval" : [0,10000],

        "source" : {
            "base_folder" : "/u40/zhanr110/MTA_ext_short/test",
            # "base_folder" : "/Users/nolanzhang/Projects/mtmct/data/MTA_ext_short/test",
            "cam_ids" : [0,1,2,3,4,5]
        }


    },

    "tracker" : {
        "type" : "DeepSort",
        "nn_budget" : 100

    }
}

