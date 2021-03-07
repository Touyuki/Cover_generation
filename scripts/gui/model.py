import argparse
import json
import math
import cv2
import os
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import numpy as np
from SRnet.datagen import datagen_srnet, example_dataset, To_tensor
from imageio import imwrite
import scipy.io as io
from SRnet.sr_model import sr_Generator
import scene_generation.vis as vis
from scene_generation.data.utils import imagenet_deprocess_batch
from scene_generation.model import Model
###1
from scene_generation.title_model2 import Generator
import torchvision.transforms.functional as F
import time
from scene_generation.util.data_load import Data_load
from scene_generation.csa_inpainting.models import create_model
import torch
import os
import torchvision
from torch.utils import data
import torchvision.transforms as transforms

# dit={"0":0,"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"H":8,"I":9,"J":10,"K":11,"L":12,"M":13,"N":14,"O":15,"P":16,"Q":17,"R":18,"S":19,"T":20,"U":21,"V":22,"W":23,"X":24,"Y":25,"Z":26}
# dic={ 0 :"0",1: "A",2: "B",3: "C",4: "D",5: "E",6:"F",7:"G",8:"H",9:"I",10:"J",11:"K",12:"L",13:"M",14:"N",15:"O",16:"P",17:"Q",18:"R",19:"S",20:"T",21:"U",22:"V",23:"W",24:"X",25:"Y",26:"Z"}

##CSA
# class Opion():
#     def __init__(self):
#         self.dataroot = "images/"  # image dataroot ##
#         self.maskroot = "mask/"  # mask dataroot  ##
#         self.batchSize = 1  # Need to be set to 1
#         self.fineSize = 256  # image size
#         self.input_nc = 3  # input channel size for first stage
#         self.input_nc_g = 6  # input channel size for second stage
#         self.output_nc = 3  # output channel size
#         self.ngf = 64  # inner channel
#         self.ndf = 64  # inner channel
#         self.which_model_netD = 'basic'  # patch discriminator
#
#         self.which_model_netF = 'feature'  # feature patch discriminator
#         self.which_model_netG = 'unet_csa'  # seconde stage network
#         self.which_model_netP = 'unet_256'  # first stage network
#         self.triple_weight = 1
#         self.name = 'csa'
#         self.n_layers_D = '3'  # network depth
#         self.gpu_ids = [0]
#         self.model = 'csa_net'
#         self.checkpoints_dir = "models/"  #  ##
#         self.norm = 'instance'
#         self.fixed_mask = 1
#         self.use_dropout = False
#         self.init_type = 'normal'
#         self.mask_type = 'random'  ##
#         self.lambda_A = 100
#         self.threshold = 5 / 16.0
#         self.stride = 1
#         self.shift_sz = 1  # size of feature patch
#         self.mask_thred = 1
#         self.bottleneck = 512
#         self.gp_lambda = 10.0
#         self.ncritic = 5
#         self.constrain = 'MSE'
#         self.strength = 1
#         self.init_gain = 0.02
#         self.cosis = 1
#         self.gan_type = 'lsgan'
#         self.gan_weight = 0.2
#         self.overlap = 4
#         self.skip = 0
#         self.display_freq = 1000
#         self.print_freq = 50
#         self.save_latest_freq = 5000
#         self.save_epoch_freq = 2
#         self.continue_train = False
#         self.epoch_count = 1
#         self.phase = 'train'
#         self.which_epoch = ''
#         self.niter = 20
#         self.niter_decay = 100
#         self.beta1 = 0.5
#         self.lr = 0.0002
#         self.lr_policy = 'lambda'
#         self.lr_decay_iters = 50
#         self.isTrain = True



# def str_to_one_hot(str):
#     lens=len(str)
#     aa=torch.zeros(lens,189)
#     # aa=torch.zeros(lens,42)
#     for i in range(0,lens):
#       new_t_vecs = torch.zeros(7, 27)
#       # new_t_vecs = torch.zeros(7, 6)
#       for j in range(0,7):
#         new_t_vecs[j][dit[str[i][j]]]=1
#         aa1=new_t_vecs.view(1,-1)
#         aa[i]=aa1
#     return aa


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output_dir', default='outputs')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])
args = parser.parse_args()


def get_model():
    if not os.path.isfile(args.checkpoint):
        print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
        print('Maybe you forgot to download pretraind models? Try running:')
        print('bash scripts/download_models.sh')
        return

    output_dir = os.path.join('scripts', 'gui', 'images', args.output_dir)
    if not os.path.isdir(output_dir):
        print('Output directory "%s" does not exist; creating it' % args.output_dir)
        os.makedirs(output_dir)

    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'gpu':
        device = torch.device('cuda:0')
        if not torch.cuda.is_available():
            print('WARNING: CUDA not available; falling back to CPU')
            device = torch.device('cpu')

    # Load the model, with a bit of care in case there are no GPUs
    map_location = 'cpu' if device == torch.device('cpu') else None
    checkpoint = torch.load(args.checkpoint, map_location=map_location)
    dirname = os.path.dirname(args.checkpoint)
    features_path = os.path.join(dirname, 'features_clustered_100.npy')
    features_path_one = os.path.join(dirname, 'features_clustered_001.npy')
    features = np.load(features_path, allow_pickle=True).item()
    features_one = np.load(features_path_one, allow_pickle=True).item()

    model = Model(**checkpoint['model_kwargs'])
    model_state = checkpoint['model_state']
    model.load_state_dict(model_state)
    model.features = features
    model.features_one = features_one
    print("features len",len(features))
    print("features_one len",len(features_one))

    model.colors = torch.randint(0, 256, [174, 3]).float()
    model.colors[0, :] = 256
    model.eval()
    model.to(device)

    return model


def js_split(json_str):
    scene = json.loads(json_str) ##载入返回json字符串
    scene = scene['objects'] ##只获取objects部分的字符串
    # json_design=[]
    json_obj=[]
    for i in scene:
        # if i["text"] in ["triangle","rectangular"]:
        #     json_design.append(i)
        # else:
            json_obj.append(i)
    # return json_design,json_obj
    return json_obj


def json_to_img(scene_graph, model):
    scene_graph=scene_graph.replace("solid","mask")
    output_dir = args.output_dir
    print("scene_graph",scene_graph)
    json_obj=js_split(scene_graph)
    # print("json_design",json_design)
    # print("json_obj",json_obj)

    scene_graphs_old = json_to_scene_graph(scene_graph,json_obj) ##获得每个物体的大小 位置 相对关系等信息(返回格式为json串)
    one_flg = True


    title = scene_graphs_old[0]['title']
    title_str=title
    title = [title]
    # print("tit_loc",tit_loc)
    scene_graphs_old[0].pop("title")
    scene_graphs=scene_graphs_old

    current_time = datetime.now().strftime('%H-%M-%S') ##


    with torch.no_grad():
                (imgs, boxes_pred, masks_pred, layout, layout_pred, _,tit_box), objs = model.forward_json(scene_graphs)## layout_pred为生成最终图像的layout图像

    imgs = imagenet_deprocess_batch(imgs)

    ####  use the mask

    # opt = Opion()
    # transform_mask = transforms.Compose(
    #     [transforms.Resize((opt.fineSize, opt.fineSize)),
    #      transforms.ToTensor(),
    #      ])
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((opt.fineSize, opt.fineSize)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    # model = create_model(opt) ##
    # load_epoch = 120
    # model.load(load_epoch)
    # dataset_test = Data_load(opt.dataroot, opt.maskroot, transform, transform_mask)
    # iterator_test = (data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=True))
    # for _, mask in (iterator_test):
    #     print("mask type",type(mask))
    #     print("mask size",mask.size())
    #     npm = np.array(mask)
    #     np.savetxt("m.txt", npm[0][0])
    #
    #     # print("imgs.size()",imgs.size())
    #     # print("type(imgs[0])",type(imgs[0]))
    #     # exit(0)
    #
    #     # img_pil=transforms.ToPILImage()(imgs[0].cpu()).convert('RGB')
    #
    #     image = imgs[0] #
    #     npi = np.array(image)
    #     np.savetxt("i.txt", npi[0][0])
    #     # image = img_pil.cuda()
    #     # image = torch.unsqueeze(img_pil, 0)
    #     # print("img_size",img_pil.size())
    #     # print("type(image)", type(img_pil))
    #     # image = transforms.ToPILImage()(image.cpu()).convert('RGB')
    #     # image=transform(img_pil)
    #     # image=transform(image.convert('RGB'))
    #     # image=transforms.ToPILImage()(image.cpu()).convert("RGB")
    #     # print("img_size",image.size())
    #     # exit()
    #     image = torch.unsqueeze(image, 0)
    #     # image=image.cuda()
    #     # image = torch.unsqueeze(image, 0)
    #     mask=mask
    #     mask = mask.cuda()
    #     mask = mask[0][0]
    #     mask = torch.unsqueeze(mask, 0)
    #     mask = torch.unsqueeze(mask, 1)
    #     mask = mask.byte()
    #     model.set_input(image/255, mask)
    #     # model.set_input(image, mask)
    #
    #     model.set_gt_latent()
    #     model.test()
    #     real_A, real_B, fake_B = model.get_current_visuals()
    #     # pic_M = (torch.cat([real_A, real_B, fake_B], dim=0) + 1) / 2.0 ##
    #     pic_M = torch.cat([real_A, real_B, fake_B], dim=0) + 1
    #
    #     # pic_M = (torch.cat([fake_B], dim=0) + 1) / 2.0
    #     torchvision.utils.save_image(pic_M, '%s/Epoch_(%d)_(%dof%d).jpg' % (
    #         "save_dir/", 5, 6, len(dataset_test)), nrow=1)
    #     print("pic_M_size",pic_M.size())
    #     np2 = np.array(pic_M[0][0].cpu())
    #     np.savetxt("2.txt", np2)
    #
    #     pic_M=pic_M*255
    #
    #     pic_M=pic_M[2]
        # print("pic_M_size2",pic_M.size())

        # exit()


    device = torch.device("cuda" if (torch.cuda.is_available() ) else "cpu")

    title_dir = "64_title_image/t/"
    title_ori_dir="64_title_image/ori/"
    title_out_dir="64_title_image/output/"

    tit_flg=False
    if len(tit_box)>1:
        print("wront title number!")
    if len(tit_box)==1:
        tit_flg=True
    if tit_flg:
        tmp_x0,tmp_x1,tmp_y0,tmp_y1=tit_box[0]
        tit_x0,tit_y0,tit_x1,tit_y1=int(tmp_x0*128),int(tmp_x1*128),int(tmp_y0*128),int(tmp_y1*128)
        tit_w,tit_h=tit_x1-tit_x0,tit_y1-tit_y0
        tit_img=imgs[0].cpu().numpy().transpose(1, 2, 0).astype('uint8')[tit_y0:tit_y1,tit_x0:tit_x1]
        cv2.imwrite(title_dir+title_str + ".png", tit_img)

    ###2 generate title image by SRnet

        sr_checkpoint="models/SRnet/trained_final_5M_.model"
        netG3=sr_Generator(in_channels=3).to(device)
        checkpoint = torch.load(sr_checkpoint)
        netG3.load_state_dict(checkpoint['generator'])
        trfms = To_tensor()
        ## read the style img and title img
        example_data = example_dataset(title_str=title_str, data_dir="title_image", transform = trfms) ##  title image dir, style image dir
        example_loader = DataLoader(dataset = example_data, batch_size = 1, shuffle = False)
        example_iter = iter(example_loader)

        netG3.eval()
        with torch.no_grad():
            for step in range(1):
                try:
                    inp = example_iter.next()

                except StopIteration:

                    example_iter = iter(example_loader)
                    inp = example_iter.next()

                i_t = inp[0].to(device)
                i_s = inp[1].to(device)

                print("i_t",i_t.shape)
                print("i_s",i_s.shape)
                # name = str(inp[2][0])

                o_sk, o_t, o_b, o_f = netG3(i_t, i_s, (i_t.shape[2], i_t.shape[3]))
                o_f = o_f.squeeze(0).detach().to('cpu')
                o_f = F.to_pil_image((o_f + 1) / 2)
                pic=cv2.cvtColor(np.asarray(o_f),cv2.COLOR_RGB2BGR)

                cv2.imwrite(title_out_dir + title_str + ".png", pic)

                break

    pic = cv2.resize(pic, (tit_w, tit_h), interpolation=cv2.INTER_CUBIC)

    for i in range(imgs.shape[0]):
        img_np = imgs[0].cpu().numpy().transpose(1, 2, 0).astype('uint8')



        if tit_flg:
                for y in range(tit_y0,tit_y0+tit_h-2):
                    for x in range(tit_x0,tit_x0+tit_w-2):
                        # if pic[y,x,i2]<130:
                        #    img_np[y,64+x,i2]=pic[y,x,i2]
                        # if int(pic[y,x,0])+int(pic[y,x,1])+int(pic[y,x,2]) < 600:
                            img_np[y,x, 0] = pic[y-tit_y0,x-tit_x0,0]
                            img_np[y,x, 1] = pic[y-tit_y0,x-tit_x0,1]
                            img_np[y,x, 2] = pic[y-tit_y0,x-tit_x0,2]

        img_path = os.path.join('scripts', 'gui', 'images', output_dir, 'img{}.png'.format(current_time))
        imwrite(img_path, img_np)
        return_img_path = os.path.join('images', output_dir, 'img{}.png'.format(current_time))
    print("")
    # Save the generated layout image
    for i in range(imgs.shape[0]):

        img_layout_np = one_hot_to_rgb(layout_pred[:, :174, :, :], model.colors)[0].numpy().transpose(1, 2, 0).astype(
            'uint8')

        obj_colors = []

        if one_flg:
            print("one")
            for obj in objs[:-1]:
                new_color = torch.cat([model.colors[obj] / 256, torch.ones(1)])
                obj_colors.append(new_color)

        img_layout_path = os.path.join('scripts', 'gui', 'images', output_dir, 'img_layout{}.png'.format(current_time))
        if one_flg:
            vis.add_boxes_to_layout(img_layout_np, scene_graphs[i]['objects'], boxes_pred, img_layout_path,
                                colors=obj_colors)
        return_img_layout_path = os.path.join('images', output_dir, 'img_layout{}.png'.format(current_time))

    # Draw and save the scene graph
    if args.draw_scene_graphs:
        for i, sg in enumerate(scene_graphs):
            sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'])
            sg_img_path = os.path.join('scripts', 'gui', 'images', output_dir, 'sg{}.png'.format(current_time))
            imwrite(sg_img_path, sg_img)
            sg_img_path = os.path.join('images', output_dir, 'sg{}.png'.format(current_time))

    return return_img_path, return_img_layout_path


def one_hot_to_rgb(one_hot, colors):
    print("size:one_hot,colors",one_hot.size(),colors.size()) ##
    one_hot_3d = torch.einsum('abcd,be->aecd', (one_hot.cpu(), colors.cpu()))
    one_hot_3d *= (255.0 / one_hot_3d.max())
    return one_hot_3d


def json_to_scene_graph(json_text,json_design):
    scene = json.loads(json_text)
    if len(scene) == 0:
        return []
    image_id = scene['image_id']
    title = scene['title']
    # scene = scene['objects']###
    scene = json_design

    print("title",title)
    ####1
    new_scene = []
    title_scene = []
    for i in range(0, len(scene)):
        # if scene[i]["text"] != "title":
            new_scene.append(scene[i])
        # else:
        #     title_scene.append(scene[i])
    # if len(title_scene) > 0:
    #     title_loc = title_scene[0]["location"] // 5  ###标题位置
    # else:
    #     title_loc = -1



###1    objects = [i['text'] for i in scene]
    objects = [i['text'] for i in new_scene]

    relationships = []
    size = []
    location = []
    features = []
    for i in range(0, len(objects)):


###1        obj_s = scene[i]
        obj_s = new_scene[i]

        # Check for inside / surrounding

        sx0 = obj_s['left']
        sy0 = obj_s['top']
        sx1 = obj_s['width'] + sx0
        sy1 = obj_s['height'] + sy0

        margin = (obj_s['size'] + 1) / 10 / 2
        mean_x_s = 0.5 * (sx0 + sx1)
        mean_y_s = 0.5 * (sy0 + sy1)

        sx0 = max(0, mean_x_s - margin)
        sx1 = min(1, mean_x_s + margin)
        sy0 = max(0, mean_y_s - margin)
        sy1 = min(1, mean_y_s + margin)

        size.append(obj_s['size'])
        location.append(obj_s['location'])

        features.append(obj_s['feature'])
        if i == len(objects) - 1:
            continue

        obj_o = new_scene[i + 1]



        ox0 = obj_o['left']
        oy0 = obj_o['top']
        ox1 = obj_o['width'] + ox0
        oy1 = obj_o['height'] + oy0

        mean_x_o = 0.5 * (ox0 + ox1)
        mean_y_o = 0.5 * (oy0 + oy1)
        d_x = mean_x_s - mean_x_o
        d_y = mean_y_s - mean_y_o
        theta = math.atan2(d_y, d_x)

        margin = (obj_o['size'] + 1) / 10 / 2
        ox0 = max(0, mean_x_o - margin)
        ox1 = min(1, mean_x_o + margin)
        oy0 = max(0, mean_y_o - margin)
        oy1 = min(1, mean_y_o + margin)

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

        relationships.append([i, p, i + 1])

    ###1 return [{'objects': objects, 'relationships': relationships, 'attributes': {'size': size, 'location': location},
         ###1    'features': features, 'image_id': image_id}]
    return [{'objects': objects, 'relationships': relationships, 'attributes': {'size': size, 'location': location},
             'features': features, 'image_id': image_id,'title':title}]