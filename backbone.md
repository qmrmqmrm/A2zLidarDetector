## backbone

resnet 50

``` python
class ResNet50:
    '''
    input 
    	batch_image : (b,3,h,w)
    
    output
		backbone_s2 : (b,256, h/4, w/4)
        backbone_s3 : (b,512, h/8, w/8)
        backbone_s4 : (b,1024, h/16, w/16)
		
    
    
    '''

```

## neck

fpn

``` python

class FPN:
    '''
    input 
    	backbone_s2 : (b,256, h/4, w/4)
        backbone_s3 : (b,512, h/8, w/8)
        backbone_s4 : (b,1024, h/16, w/16)
        
    output
    	neck_s2 : (b,256, h/4, w/4)
        neck_s3 : (b,256, h/8, w/8)
        neck_s4 : (b,256, h/16, w/16)
    '''
```



## rpn

``` python
class RPN:
    '''
    input
    	neck_s2 : (b,256, h/4, w/4)
        neck_s3 : (b,256, h/8, w/8)
        neck_s4 : (b,256, h/16, w/16)
    	gt
    output
    	
		rpn_bbox2d
		rpn_bbox2d_logit : (batch, Nhd, 4)  -> padding
		rpn_object_logit : (batch, Nhd, 1)  -> padding
		rpn_proposal :
			rpn_bbox2d_logit : (batch, N, 4)
			rpn_object_logit : (batch, N, 1)
    '''
```



## Head

``` python
class Head:
    '''
    input
    	rpn_bbox2d_logit : (batch, Nhd, 4)
    output
    	head_bbox3d_logit : (batch, Nhd, C, 6)
    	head_yaw_resi : (batch, Nhd, C, B)
    	head_yaw_logit : (batch, Nhd, C, B)
    	head_category_logit: (batch, Nhd, C)
    	
    '''
```



## loss

common / auxi

``` python
def make_auxi():
    '''
    input
    	gt
    	pred
    output
    	auxi
    	“gt_aligned”: {“bbox2d”: [batch, Nhd,4],
                        “bbox3d”: [batch, Nhd,6],
                        “object”: [batch, Nhd,1],
                        “yaw”: [batch, Nhd,1]
                        "category":[batch, Nhd, 1]
                        },
        “gt_feature”:  {“bbox2d”: (batch, N, 4)
        				"bbox2d_logit": (batch, N, 4)
                        “bbox3d”: (batch, N, 6)
                        “object”: (batch, N, 1)
                        “yaw”: (batch, N, 1)
                        "category":[batch, N, 1]

    '''
    
def make_align(gt):
    '''
    input 
     	gt (batch,n,c)
	output
		gt_align [batch,Nhd,c]

	'''
def make_feature(gt, anchor):
	'''
	input
		gt (batch,n,c)
		anchor (batch, N, 4)
	output
        gt_feature (batch, N, c)
        
    '''
    
def select_class_from_prediction():
    '''
    input
    	pred :
            head_bbox3d_logit : (batch, Nhd, C, 6)
            head_yaw_resi : (batch, Nhd, C, B)
            head_yaw_logit : (batch, Nhd, C, B)
            head_category_logit: (batch, Nhd, C)
            
        gt_aligned:
             {“bbox2d”: [batch, Nhd, 4],
             “bbox3d”: [batch, Nhd, 6],
             “object”: [batch, Nhd, 1],
             “yaw”: [batch, Nhd, 1],
             "category":[batch, Nhd, 1]
             },
        	
    output:
    	selcet_pred:
            head_bbox3d_logit : (batch, Nhd,  6)
            head_yaw_resi : (batch, Nhd, 1)
            head_yaw_logit : (batch, Nhd,  1)
            head_category_logit: (batch, Nhd, 1)
    '''
```



loss

``` python
class Box2dRegression:
    def __call__():
        '''
        input
        	rpn_bbox2d_logit (batch, N, 4)
        	gt_bbox2d_logit_feature (batch, N, 4)
        
      	output
      		huber loss(rpn_bbox2d_logit, gt_bbox2d_logit_feature)
        '''

class Objectness:
    def __call__():
        '''
        input
        	rpn_object_logit (batch, N, 1)
        	gt_object_feature (batch,N, 1)
        output
        	binary cross entropy(rpn_object_logit, gt_object_feature)
        
        '''
class Box3dRegression:
    def __call__():
        '''
        input
        	head_bbox3d_sel (batch, Nhd, 6)
        	gt_bbox3d_logit (batch, Nhd, 6)
        output
        	huber(head_bbox3d_sel, gt_bbox3d_logit)
        	
        '''
        
class YawRegression:
    def __call__():
        '''
        input
        	head_yaw_resi_sel (batch, Nhd, 1)
        	gt_yaw_resi_aligned (baych, Nhd,1)
        output
        	huber(head_yaw_resi_sel, gt_yaw_resi_aligned)
        
        
        '''
class Category():
    def __call__():
        '''
        input
        	head_category_logit (batch, Nhd, C)
        	gt_category_aligned (btach, Nhd, 1)
        output
        	cross_entropy(head_category_logit, gt_category_aligned)
        '''
        
class YawCategory():
    def __call__():
        """
        input
        	head_yaw_logit_seL (batch, Nhd, B)
        	gt_yaw_category_aligned ( batych, Nhd, 1)
       	output
       		cross_entropy(head_yaw_logit_seL, gt_yaw_category_aligned)
        """
        
```





## logger

``` python
class history_logger:
    
    '''
    
    input
    
    output
    	
    
    
    '''



```

