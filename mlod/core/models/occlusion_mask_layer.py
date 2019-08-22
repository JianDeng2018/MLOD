import tensorflow as tf

class OccMaskLayer():
    def build(self,depth_input,boxes,box_indices,ref_depth_min,ref_depth_max,n_split,mask_size,img_size,mask_quantile_level):
        """
        masking the feature, if the median of depth_map in the bounding box is less than the ref_depth_min, and larger than ref_depth_max
        img_input: H * W * C
        depth_input: H * W
        boxes_norm: [num_boxes, 4]
        n_split: int that divides H and W
        ref_height: [num_boxes, depth]
        """
        with tf.variable_scope("occ_mask"):
            self.n_split = n_split
            #self.n_batch = tf.cast(boxes.shape[0],tf.int32)
            self.n_batch = tf.shape(boxes)[0]
            sub_box = self.slice_box_gen(boxes)
            #duplicate the reference depth
            ref_depth_min = tf.expand_dims(ref_depth_min,-1)
            ref_depth_min_dup = tf.tile(ref_depth_min,[1,self.n_split**2])
            ref_depth_min_dup = tf.reshape(ref_depth_min_dup,[self.n_batch*self.n_split**2,1],name='duplicated_depth_min')

            ref_depth_max = tf.expand_dims(ref_depth_max,-1)
            ref_depth_max_dup = tf.tile(ref_depth_max,[1,self.n_split**2])
            ref_depth_max_dup = tf.reshape(ref_depth_max_dup,[self.n_batch*self.n_split**2,1],name='duplicated_depth_max')

            #duplicate the reference depth
            box_indices = tf.expand_dims(box_indices,-1)
            box_indices_dup = tf.tile(box_indices,[1,self.n_split**2])
            box_indices_dup = tf.reshape(box_indices_dup,[self.n_batch*self.n_split**2],name='duplicated_box_indices')

            # must use nearest neighbour method
            depth_size = mask_size[0]*mask_size[1]
            crop_depth = tf.image.crop_and_resize(
                    depth_input,
                    sub_box,
                    box_indices_dup,
                    mask_size,method='nearest')
            crop_depth = tf.reshape(crop_depth,[self.n_batch*self.n_split**2, depth_size])
            #map_params = (crop_depth,ref_depth_dup)
            
            method = 'median'
            #if method == 'median':
            #medidan
            # avoid empty case
            fill_in = tf.tile([[0.1,100.0]],[self.n_batch*self.n_split**2,1])
            crop_depth = tf.concat([crop_depth,fill_in],axis = 1)

            num_nonzero = tf.count_nonzero(crop_depth,axis = 1)
            # roll out value = 0 and calculate the median of the rest
            quantile_idx = tf.ceil(depth_size - tf.cast(num_nonzero,dtype=tf.float32)*mask_quantile_level)
            quantile_idx = tf.cast(quantile_idx, dtype=tf.int32)
            quantile_idx = tf.expand_dims(quantile_idx,-1) 
            batch_range = tf.expand_dims(tf.range(0, self.n_batch*self.n_split**2),-1)
            cat_idx = tf.concat([batch_range, quantile_idx], axis=1)
            
            sorted_crop_depth = tf.contrib.framework.sort(crop_depth, axis= 1)
            depth_val = tf.gather_nd(sorted_crop_depth,cat_idx)
            
            """
            f(x) = 1  if x_min <= x <= x_max
                   0  otherwise

            f(x) = g1(x,x_min) + g(x_max,x) - 1

            where g(x, y) = 1   if x >= y
                            0   otherwise
            """
            occ = self.step_f(depth_val, tf.squeeze(ref_depth_min_dup,1))+self.step_f(tf.squeeze(ref_depth_max_dup,1), depth_val)-1
            dep_zero = tf.cast(tf.less(depth_val, 0.2), dtype=tf.float32)
            occ +=dep_zero
            
            occ = tf.expand_dims(occ,0)
            occ = tf.reshape(occ,[self.n_batch,self.n_split*self.n_split],name='occ_mask_base')
            num_masks = tf.count_nonzero(occ,axis=1)
            mask_weights = (tf.cast(num_masks,tf.float32)+0.01)/(self.n_split*self.n_split)
            occ = occ/tf.expand_dims(mask_weights,-1)
            occ = tf.reshape(occ,[self.n_batch,self.n_split,self.n_split],name='occ_mask_base')
            occ = tf.expand_dims(occ,-1)
            occ_mask = tf.image.resize_nearest_neighbor(occ,img_size,name='occ_mask')
            #occ_mask = tf.squeeze(occ_mask,axis=-1,name='occ_mask')
            return occ_mask

    def slice_box_gen(self,boxes):
        #boxes = tf.cast(boxes,tf.int32)
        print('boxes',boxes)
        y2 = tf.reshape(boxes[:,2],(self.n_batch,1))
        y1 = tf.reshape(boxes[:,0],(self.n_batch,1))
        x1 = tf.reshape(boxes[:,1],(self.n_batch,1))
        
        x2 = tf.reshape(boxes[:,3],(self.n_batch,1))

        y_split_size = (y2-y1)/self.n_split
        x_split_size = (x2-x1)/self.n_split

        i = tf.cast(tf.range(self.n_split),tf.float32)
        i = tf.expand_dims(i,0)
        i2 = tf.tile(tf.transpose(i,[1,0]),[1,self.n_split])
        i2 = tf.reshape(i2,[1,self.n_split**2]) 

        x_i =tf.tile(tf.matmul(x_split_size,i),[1,self.n_split] ) 
        y_i = tf.matmul(y_split_size,i2)

        ya = tf.reshape(y1+y_i,((self.n_split**2)*self.n_batch,1))
        xa = tf.reshape(x1+x_i,((self.n_split**2)*self.n_batch,1))

        x_size = tf.tile(x_split_size,[1,self.n_split**2])
        x_size = tf.reshape(x_size,[self.n_batch*self.n_split**2,1])

        y_size = tf.tile(y_split_size,[1,self.n_split**2])
        y_size = tf.reshape(y_size,[self.n_batch*self.n_split**2,1])

        yb = ya + y_size
        xb = xa + x_size

        sub_box = tf.concat([ya,xa,yb,xb],axis = 1,name='sub_boxes')

        return sub_box

    def step_f(self,x, y):
        """
        g(x, y) = 1   if x >= y
                  0   otherwise
        """
        return (tf.sign(x - y)+1)/2.0







