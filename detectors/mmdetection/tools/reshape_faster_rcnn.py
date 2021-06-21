import torch


pretrained_weights  = torch.load('/home/koehlp/Downloads/work_dirs/detector/faster_rcnn_x101/faster_rcnn_x101_64x4d_fpn_1x_20181218-c9c69c8f.pth')

num_class = 2
pretrained_weights['state_dict']['bbox_head.fc_cls.weight'].resize_(num_class, 1024)
pretrained_weights['state_dict']['bbox_head.fc_cls.bias'].resize_(num_class)
pretrained_weights['state_dict']['bbox_head.fc_reg.weight'].resize_(num_class*4, 1024)
pretrained_weights['state_dict']['bbox_head.fc_reg.bias'].resize_(num_class*4)

#You still need to add a hash
torch.save(pretrained_weights, "/home/koehlp/Downloads/work_dirs/detector/faster_rcnn_x101/faster_rcnn_x101_64x4d_fpn_1x_20181218-c9c69c8f_2cls.pth")