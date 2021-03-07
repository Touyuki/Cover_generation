import json
import math
import os
import pickle
import random
from collections import defaultdict

import PIL
import numpy as np
import pycocotools.mask as mask_utils
import torch
import torchvision.transforms as T
from skimage.transform import resize as imresize
from torch.utils.data import Dataset

from .utils import imagenet_preprocess, Resize

PREDICATES_VALUES = ['left of', 'right of', 'above', 'below', 'inside', 'surrounding']
# PREDICATES_VALUES = ['above',  'inside', 'surrounding','below'] ##


class CocoSceneGraphDataset(Dataset):
    def __init__(self, image_dir, instances_json, stuff_json=None, stuff_only=True, image_size=(64, 64), mask_size=16,
                 normalize_images=True, max_samples=None, min_object_size=0.02,
                 min_objects_per_image=4, max_objects_per_image=8, include_other=False, instance_whitelist=None,
                 stuff_whitelist=None, no__img__=False, sample_attributes=False, test_part=False, size_attribute_len=10,
                 grid_size=25): ##max_objects_per_image old 8  ##stuff_only=True
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        Inputs:
        - image_dir: Path to a directory where images are held
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - image_size: Size (H, W) at which to load images. Default (64, 64).
        - mask_size: Size M for object segmentation masks; default 16.
        - normalize_image: If True then normalize images by subtracting ImageNet
          mean pixel and dividing by ImageNet std pixel.
        - max_samples: If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist: None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
        """
        super(Dataset, self).__init__()

        if stuff_only and stuff_json is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')

        self.image_dir = image_dir
        self.mask_size = mask_size
        self.max_samples = max_samples
        self.normalize_images = normalize_images
        self.set_image_size(image_size)
        self.no__img__ = no__img__

        with open(instances_json, 'r',encoding="utf-8-sig") as f: ##
            instances_data = json.load(f)

        self.image_id_to_sentences = {}
        stuff_data = None
        if stuff_json is not None and stuff_json != '':
            with open(stuff_json, 'r',encoding="utf-8-sig") as f:  ##
                stuff_data = json.load(f)

        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }
        self.size_attribute_len = size_attribute_len
        self.location_attribute_len = grid_size
        self.vocab['num_attributes'] = self.size_attribute_len + self.location_attribute_len
        object_idx_to_name = {}
        all_instance_categories = []
        for category_data in instances_data['categories']:
            category_id = category_data['id']
            category_name = category_data['name']
            all_instance_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id
        all_stuff_categories = []
        if stuff_data:
            for category_data in stuff_data['categories']:
                category_name = category_data['name']
                category_id = category_data['id']
                all_stuff_categories.append(category_name)
                object_idx_to_name[category_id] = category_name
                self.vocab['object_name_to_idx'][category_name] = category_id

        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        if stuff_whitelist is None:
            stuff_whitelist = all_stuff_categories
        self.vocab['instance'] = instance_whitelist
        self.vocab['stuff'] = stuff_whitelist
        category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

        # Add object data from instances
        self.image_id_to_objects = defaultdict(list)
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            # W, H = self.image_id_to_size[image_id]
            try:
                W, H = self.image_id_to_size[image_id]
            except Exception as e:
                pass
                continue
            box_area = (w * h) / (W * H)
            box_ok = box_area > min_object_size
            object_name = object_idx_to_name[object_data['category_id']]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other
            if box_ok and category_ok and other_ok:
                self.image_id_to_objects[image_id].append(object_data)

        # Add object data from stuff
        if stuff_data:
            image_ids_with_stuff = set()
            ttt=1
            for object_data in stuff_data['annotations']:
                image_id = object_data['image_id']
                _, _, w, h = object_data['bbox']
                try:
                    W, H = self.image_id_to_size[image_id]
                except Exception as e:
                    pass
                    continue
                # W, H = self.image_id_to_size[image_id]
                # print("111111",W)

                image_ids_with_stuff.add(image_id)
                box_area = (w * h) / (W * H)
                box_ok = box_area > min_object_size
                object_name = object_idx_to_name[object_data['category_id']]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok:
                    self.image_id_to_objects[image_id].append(object_data)
            if stuff_only:
                new_image_ids = []
                for image_id in self.image_ids:
                    if image_id in image_ids_with_stuff:
                        new_image_ids.append(image_id)
                self.image_ids = new_image_ids

                all_image_ids = set(self.image_id_to_filename.keys())
                image_ids_to_remove = all_image_ids - image_ids_with_stuff
                for image_id in image_ids_to_remove:
                    self.image_id_to_filename.pop(image_id, None)
                    self.image_id_to_size.pop(image_id, None)
                    self.image_id_to_objects.pop(image_id, None)

        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['object_name_to_idx']['__image__'] = 0

        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name

        # Prune images that have too few or too many objects
        new_image_ids = []
        total_objs = 0
        print("min_objects_per_image", min_objects_per_image, "max_objects_per_image", max_objects_per_image)

        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if min_objects_per_image <= num_objs <= max_objects_per_image:
                    new_image_ids.append(image_id)
        self.image_ids = new_image_ids
        if test_part:
            self.image_ids = self.image_ids[1024:]

        objects_map = set()
        for image_id in self.image_ids:
            for object in self.image_id_to_objects[image_id]:
                object_class = object['category_id']
                objects_map.add(object_class)

        objects_map = list(objects_map)
        object_to_idx = {v: k + 1 for k, v in enumerate(objects_map)}
        object_to_idx[0] = 0
        self.object_to_idx = object_to_idx
        self.idx_to_object = {v: k for k, v in object_to_idx.items()}
        self.vocab['object_to_idx'] = object_to_idx
        self.vocab['my_idx_to_obj'] = [self.vocab['object_idx_to_name'][i] for i in objects_map]
        self.object_num = len(object_to_idx)

        self.vocab['pred_idx_to_name'] = ['__in_image__'] + PREDICATES_VALUES
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx





###2
        # restore_path = 'models/128/checkpoint_with_model.pt'
        # tmp_check=torch.load(restore_path)
        # tmp_vo=tmp_check["vocab"]
        # tmp_vo['object_name_to_idx']['Title'] = 190
        # tmp_vo['stuff'].append('Title')
        # tmp_vo['object_idx_to_name'].append('Title')
        # tmp_vo['object_to_idx']['190'] = 172
        # tmp_vo['my_idx_to_obj'].append('Title')
        # self.vocab=tmp_vo



        self.sample_attributes = None
        if sample_attributes:
            with open('./models/attributes_{}_{}.pickle'.format(self.size_attribute_len, self.location_attribute_len),
                      'rb') as f:
                self.sample_attributes = pickle.load(f)

    def insert_pre_trained_vocab(self, object_to_idx):
        object_to_idx = {int(k): v for k, v in object_to_idx.items()}
        self.object_to_idx = object_to_idx
        self.vocab['object_to_idx'] = object_to_idx
        self.vocab['my_idx_to_obj'] = [None] * len(object_to_idx)
        for real_ind, my_ind in object_to_idx.items():
            self.vocab['my_idx_to_obj'][my_ind] = self.vocab['object_idx_to_name'][real_ind]

    def set_image_size(self, image_size):
        # print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        if self.normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.image_size = image_size

    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            if self.max_samples and i >= self.max_samples:
                break
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def __len__(self):
        if self.max_samples is None:
            return len(self.image_ids)
        return min(len(self.image_ids), self.max_samples)

    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 1 is object.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        image_id = self.image_ids[index]

        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        H, W = self.image_size
        objs, boxes, masks = [], [], []
        add_img = 0 if self.no__img__ else 1
        size_attribute = torch.zeros([len(self.image_id_to_objects[image_id]) + add_img, self.size_attribute_len],
                                     dtype=torch.float)
        location_attribute = torch.zeros(
            [len(self.image_id_to_objects[image_id]) + add_img, self.location_attribute_len], dtype=torch.float)
        for i, object_data in enumerate(self.image_id_to_objects[image_id]):
            objs.append(self.object_to_idx[object_data['category_id']])##2
            # objs.append(1) ##


        ##
            img_id=object_data["image_id"]

            x, y, w, h = object_data['bbox']
            x0 = x / WW
            y0 = y / HH
            x1 = (x + w) / WW
            y1 = (y + h) / HH
            boxes.append(torch.FloatTensor([x0, y0, x1, y1]))
            if self.sample_attributes is not None:
                category_distr = np.asarray(
                    self.sample_attributes['size'][self.vocab['object_idx_to_name'][object_data['category_id']]])
                category_distr = category_distr / np.sum(category_distr)
                size_index = np.random.choice(self.size_attribute_len, 1, p=category_distr)[0]
            else:
                # print("whWWHH,id",w,h,WW,HH,image_id)
                size_index = round((self.size_attribute_len - 1) * (w * h) / (WW * HH))
            # print("size_index",size_index)##
            ##
            if size_index>10:
                print("wrong size index",img_id)





            size_attribute[i, size_index] = 1.0
            # This will give a numpy array of shape (HH, WW)
            mask = seg_to_mask(object_data['segmentation'], WW, HH)






            # print("mask",mask)  ##

            # Crop the mask according to the bounding box, being careful to
            # ensure that we don't crop a zero-area region
            mx0, mx1 = int(round(x)), int(round(x + w))
            my0, my1 = int(round(y)), int(round(y + h))
            mx1 = max(mx0 + 1, mx1)
            my1 = max(my0 + 1, my1)
            # print("mx0mx1my0my1",mx0,mx1,my0,my1)
            mask = mask[my0:my1, mx0:mx1]
            # np.savetxt("mask_err/"+image_id+".txt", mask, fmt='%f', delimiter=',')
            # print("x0y0x1y1",x0,y0,x1,y1)
            # print("image_id",type(mask))##
            try:
                mask = imresize(255.0 * mask, (self.mask_size, self.mask_size), mode='constant', anti_aliasing=True)
            except:
                print("wrong  mask,id",img_id)
                objs.pop()
                boxes.pop()
                continue
            else:
                mask = torch.from_numpy((mask > 128).astype(np.int64))
                masks.append(mask)

        # Add dummy __image__ object
        if not self.no__img__:
            objs.append(self.object_to_idx[self.vocab['object_name_to_idx']['__image__']])
            size_attribute[-1, self.size_attribute_len - 1] = 1.0
            boxes.append(torch.FloatTensor([0, 0, 1, 1]))
            masks.append(torch.ones(self.mask_size, self.mask_size).long())

        objs = torch.LongTensor(objs)
        boxes = torch.stack(boxes, dim=0)
        masks = torch.stack(masks, dim=0)

        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Compute centers of all objects
        obj_centers = []
        location_distr = []
        l_root = self.location_attribute_len ** (.5)
        _, MH, MW = masks.size()
        for i, obj_idx in enumerate(objs):
            x0, y0, x1, y1 = boxes[i]
            mask = (masks[i] == 1)
            xs = torch.linspace(x0, x1, MW).view(1, MW).expand(MH, MW)
            ys = torch.linspace(y0, y1, MH).view(MH, 1).expand(MH, MW)
            if mask.sum() == 0:
                mean_x = 0.5 * (x0 + x1)
                mean_y = 0.5 * (y0 + y1)
            else:
                mean_x = xs[mask].mean()
                mean_y = ys[mask].mean()
            obj_centers.append([mean_x, mean_y])
            # obj_name = self.vocab['object_idx_to_name'][self.idx_to_object[objs[i].item()]] ##2
            obj_name = "window-blind"

            if self.sample_attributes is not None and obj_name != '__image__':
                category_distr = np.asarray(self.sample_attributes['location'][obj_name])
                location_distr.append(category_distr)
            else:
                location_index = round(mean_x.item() * (l_root - 1)) + l_root * round(mean_y.item() * (l_root - 1))
                location_attribute[i, int(location_index)] = 1.0
        obj_centers = torch.FloatTensor(obj_centers)

        # Add triples
        triples = []
        num_objs = objs.size(0)
        # print("num_objs",num_objs)
        __image__ = self.object_to_idx[self.vocab['object_name_to_idx']['__image__']]
        __gmask__ = self.object_to_idx[self.vocab['object_name_to_idx']['mask']] ## 210212
        real_objs = []
        if num_objs > 1:
            # real_objs = (objs != __image__).nonzero().squeeze(1) #210212
            # real_objs = (objs != __image__ or __gmask__).nonzero().squeeze(1)##将真正代表物体的序号取出 变为1维数组 ##排除mask的位置关系
            real_objs=[]
            real_objs1 = (objs != __image__).nonzero().squeeze(1)
            real_objs2 = (objs != __gmask__).nonzero().squeeze(1)
            for idx in range(0, len(real_objs2)):
                if (real_objs2[idx] in real_objs1):
                    real_objs.append(int(real_objs2[idx]))
        # print("len real,objects",len(real_objs))
        for cur in real_objs1:
            choices = [obj for obj in real_objs if obj != cur]
            if len(choices) == 0:
                break
            other = random.choice(choices)
            if random.random() > 0.5:
                s, o = cur, other
            else:
                s, o = other, cur

            # Check for inside / surrounding
            sx0, sy0, sx1, sy1 = boxes[s]
            ox0, oy0, ox1, oy1 = boxes[o]
            d = obj_centers[s] - obj_centers[o]
            theta = math.atan2(d[1], d[0])

            if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
                p = 'surrounding'
            elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
                p = 'inside'
            elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
                p = 'left of'
            elif -3 * math.pi / 4 <= theta < -math.pi / 4:
                p = 'above'
            elif -math.pi / 4 <= theta < math.pi / 4:
                p = 'right of'
            elif math.pi / 4 <= theta < 3 * math.pi / 4:
                p = 'below'

            # if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
            #     p = 'surrounding'
            # elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
            #     p = 'inside'
            # elif sy0 <= oy0 :
            #     p = 'above'
            # elif sy0  > oy0 :
            #     p = 'below'
            # else:
            #     p="error"
            # print("s,o",s,o)

            if self.sample_attributes is not None:
                location_index, size_index = self.get_location_and_size(s, p, o, location_attribute, size_attribute,
                                                                        location_distr)
                if location_index is not None:
                    location_attribute[s.item(), location_index] = 1.0
                if size_index is not None:
                    old, new = size_index
                    size_attribute[s.item(), old] = 0
                    size_attribute[s.item(), new] = 1.

                location_index, size_index = self.get_location_and_size(o, opposite_of(p), s, location_attribute,
                                                                        size_attribute, location_distr)
                if location_index is not None:
                    location_attribute[o.item(), location_index] = 1.0
                if size_index is not None:
                    old, new = size_index
                    size_attribute[o.item(), old] = 0
                    size_attribute[o.item(), new] = 1.
            # print("###########")
            # print(objs)
            # print(p)
            p = self.vocab['pred_name_to_idx'][p]
            triples.append([s, p, o])

        # Add __in_image__ triples
        if not self.no__img__:
            O = objs.size(0)
            in_image = self.vocab['pred_name_to_idx']['__in_image__']
            for i in range(O - 1):
                triples.append([i, in_image, O - 1])

        triples = torch.LongTensor(triples)
        attributes = torch.cat([size_attribute, location_attribute], dim=1)
        return image, objs, boxes, masks, triples, attributes

    def get_location_and_size(self, s, p, o, location_attribute, size_attribute, location_distr):
        location_index, size_index = None, None
        s_index = s.item()
        o_index = o.item()
        if torch.sum(location_attribute[s_index, :]).item() == 1:
            return location_index, size_index

        s_distr = location_distr[s_index]
        if torch.sum(location_attribute[o_index, :]).item() == 1:
            o_location = np.argwhere(location_attribute[o_index, :].numpy() == 1)[0, 0]
            if p == 'surrounding':
                o_size = np.argwhere(size_attribute[o_index, :].numpy() == 1)[0, 0]
                s_size = np.argwhere(size_attribute[s_index, :].numpy() == 1)[0, 0]
                if o_size <= s_size:
                    size_index = (int(s_size), max(0, o_size - 1))
                return o_location, size_index
            elif p == 'inside':
                o_size = np.argwhere(size_attribute[o_index, :].numpy() == 1)[0, 0]
                s_size = np.argwhere(size_attribute[s_index, :].numpy() == 1)[0, 0]
                if o_size >= s_size:
                    size_index = (int(s_size), min(size_attribute.size(1) - 1, o_size + 1))
                return o_location, size_index
            elif p == 'left of':
                if o_location % 4 <= 3:
                    s_distr[3] = s_distr[7] = s_distr[11] = s_distr[15] = 0
                if o_location % 4 <= 2:
                    s_distr[2] = s_distr[6] = s_distr[10] = s_distr[14] = 0
                if o_location % 4 <= 1:
                    s_distr[1] = s_distr[5] = s_distr[9] = s_distr[13] = 0
            elif p == 'right of':
                if o_location % 4 >= 0:
                    s_distr[0] = s_distr[4] = s_distr[8] = s_distr[12] = 0
                if o_location % 4 >= 1:
                    s_distr[1] = s_distr[5] = s_distr[9] = s_distr[13] = 0
                if o_location % 4 >= 2:
                    s_distr[2] = s_distr[6] = s_distr[10] = s_distr[14] = 0
            elif p == 'above':
                if o_location <= 15:
                    s_distr[15] = s_distr[14] = s_distr[13] = s_distr[12] = 0
                if o_location <= 11:
                    s_distr[11] = s_distr[10] = s_distr[9] = s_distr[8] = 0
                if o_location <= 7:
                    s_distr[7] = s_distr[6] = s_distr[5] = s_distr[4] = 0
            elif p == 'below':
                if o_location >= 0:
                    s_distr[0] = s_distr[1] = s_distr[2] = s_distr[3] = 0
                if o_location >= 4:
                    s_distr[4] = s_distr[5] = s_distr[6] = s_distr[7] = 0
                if o_location >= 8:
                    s_distr[8] = s_distr[9] = s_distr[10] = s_distr[11] = 0

        s_distr = s_distr / np.sum(s_distr)
        location_index = int(np.random.choice(self.location_attribute_len, 1, p=s_distr))
        return location_index, size_index


def seg_to_mask(seg, width=1.0, height=1.0):
    """
    Tiny utility for decoding segmentation masks using the pycocotools API.
    """
    if type(seg) == list:
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif type(seg['counts']) == list:
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg 
    return mask_utils.decode(rle)


def opposite_of(p):
    predicates = [
        'left of',
        'above',
        'inside',
        'surrounding',
        'below',
        'right of'
    ]
    # predicates = [
    #     'above',
    #     'inside',
    #     'surrounding',
    #     'below'
    # ]
    return predicates[6-predicates.index(p)]


def coco_collate_fn(batch):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving object categories
    - boxes: FloatTensor of shape (O, 4)
    - masks: FloatTensor of shape (O, M, M)
    - triples: LongTensor of shape (T, 3) giving triples
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    - triple_to_img: LongTensor of shape (T,) mapping triples to images
    - attributes: FloatTensor of shape (O, A)
    """
    all_imgs, all_objs, all_boxes, all_masks, all_triples = [], [], [], [], []
    all_obj_to_img, all_triple_to_img, all_attributes = [], [], []
    obj_offset = 0
    for i, (img, objs, boxes, masks, triples, attributes) in enumerate(batch):
        all_imgs.append(img[None])
        if objs.dim() == 0 or triples.dim() == 0:
            continue
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        all_masks.append(masks)
        all_attributes.append(attributes)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_masks = torch.cat(all_masks)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)
    all_attributes = torch.cat(all_attributes)

    out = (all_imgs, all_objs, all_boxes, all_masks, all_triples,
           all_obj_to_img, all_triple_to_img, all_attributes)
    return out


def coco_collate_fn_with_sentences(batch):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving object categories
    - boxes: FloatTensor of shape (O, 4)
    - masks: FloatTensor of shape (O, M, M)
    - triples: LongTensor of shape (T, 3) giving triples
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    - triple_to_img: LongTensor of shape (T,) mapping triples to images
    - attributes: FloatTensor of shape (O, A)
    """
    all_imgs, all_objs, all_boxes, all_masks, all_triples = [], [], [], [], []
    all_obj_to_img, all_triple_to_img, all_attributes = [], [], []
    obj_offset = 0
    sentences = []
    for i, (img, objs, boxes, masks, triples, attributes, sentence) in enumerate(batch):
        all_imgs.append(img[None])
        if objs.dim() == 0 or triples.dim() == 0:
            continue
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        all_masks.append(masks)
        all_attributes.append(attributes)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)
        sentences.append(sentence)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_masks = torch.cat(all_masks)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)
    all_attributes = torch.cat(all_attributes)

    out = (all_imgs, all_objs, all_boxes, all_masks, all_triples,
           all_obj_to_img, all_triple_to_img, all_attributes, sentences)
    return out
