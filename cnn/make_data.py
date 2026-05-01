import torch
import numpy as np
import sklearn
from torch.utils.data import Dataset
import math
from debug import *
from seeder import *

import matplotlib.pyplot as plt

#TBD: could move all consts to a single consts_mgr file
num_samples=1000
num_dots=100

#TBD: refactor the circle, square scatter generators into a separate util file
CIRCLE=0
RECTANGLE=1

def display_images(images):
    if (checkVerbosity(INFO)):
        for i in range(images.shape[2]):
            plt.imshow(images[:, :, i].numpy(), cmap='gray')
            plt.title(f"Sample {i}")
            plt.show()
            inp = input("Press Enter for next, 'q' to quit: ")
            if inp.strip().lower() == 'q':
                break

def make_scatter_square():
    # Thoughts: get a random left top. Orientation: (0,0) is the overall most left,top
    # random height, width. 
    # constraint: clamp height and width if it crosses past the edge top + height, left + width < 28.
    # Total perimeter is (height + width) * 2. Generate many points at different points along the perimeter starting from the left top
    # points actual coordinates. (tl - top-line, rl - right-line, bl - bottom-line, ll - left line)
          # tl: p <= width => (left+p, top); 
          # rl: p > width < width+height => left+width, top+(p-width)
          # bl: p > width+height < 2*width+height => left+(p-height-width), top + height 
          # ll: p > 2*width+height < 2*width + 2*height => left, top + (p-2*width -height)
     
    # Add a small perturbation to the x,y of the dots
    # generate num_dots and num_samples, return in a tuple format suitable for torch DataSet

    height = torch.randint(2,28, (1, num_samples))
    width = torch.randint(2,28, (1, num_samples))
    left = torch.randint(0,15, (1, num_samples))
    top = torch.randint(0,15, (1, num_samples))

    # clamp to the edge
    index = (left + width) >= 27
    width[index] = 27-left[index]  

    index = (top + height) >= 27
    height[index] = 27-top[index]  

    # rand(1) is a dim=1 size=1 tensor, should broadcast alright.
    # expand is handy
    max_p = (2 * (width + height)).expand(num_dots, num_samples)

    debug(f"max_p.shape is {max_p.shape} rand(num_dots, num_samples).shape is {torch.rand(num_dots, num_samples).shape}")

    # hadamard product to get a different p for each dot
    p = max_p * torch.rand(num_dots, num_samples)

    debug(f"p.shape is {p.shape}")
    debug("p is ", p)

    width_expanded = width.expand(num_dots, num_samples)
    height_expanded = height.expand(num_dots, num_samples)
    top_expanded = top.expand(num_dots, num_samples)
    left_expanded = left.expand(num_dots, num_samples)

    index_tl = p <= width_expanded
    index_rl = (p > width_expanded) & (p <= (width_expanded + height_expanded))
    index_bl = (p > (width_expanded + height_expanded))  & (p <= (2*width_expanded + height_expanded))
    index_ll = (p > (2*width_expanded + height_expanded)) & (p <= (2*width_expanded + 2* height_expanded))

    # shape of dots: num_dots, num_dots, num_samples
    
    sample_list = torch.arange(0, num_samples).expand(num_dots, num_samples)
    x_list = torch.zeros(num_dots, num_samples, dtype=torch.long)
    y_list = torch.zeros(num_dots, num_samples, dtype=torch.long)

    # TBD: top,left,width,height should broadcast right 1, num_sampels, trailing dim matches
    # learning boolen indexing is flattening, p[index_rl] -width doesnt work. (p-width)[index_rl] does
    debug(f"index_tl.shape is {index_tl.shape}: left.shape is {left.shape}, p[index_tl].shape is {p[index_tl].shape} " )
    x_list [ index_tl ] = ((left_expanded + p)[index_tl]).long()
    y_list [ index_tl ] = top_expanded[index_tl].long() # assignment should also broadcast okay. TBD: Compile Check

    x_list [ index_rl ] = (left_expanded + width_expanded)[index_rl].long()
    y_list [ index_rl ] = (top_expanded + p - width_expanded)[index_rl].long()
    
    x_list [ index_bl ] = (left_expanded + p-height_expanded-width_expanded)[index_bl].long()
    y_list [ index_bl ] = (top_expanded + height_expanded)[index_bl].long()

    x_list [ index_ll ] = left_expanded[index_ll].long()
    y_list [ index_ll ] = (top_expanded + p - 2*width_expanded -height_expanded) [index_ll].long()

    # shape of the image : 28, 28, num_sample
    images = torch.zeros(28,28, num_samples)

    debug( "x_list", x_list)

    images [ x_list, y_list, sample_list ] = 1
 
    # Displays if verbosity is INFO
    display_images(images)

    ret_images = torch.permute(images.unsqueeze(2), (3, 2, 0, 1))
    debug (ret_images.shape)

    label = torch.full((num_samples,), RECTANGLE)
    
    return (ret_images, label)
   

    
# FIXME: circle can be recoded and simplified.
# Square was coded after circle and the square look a bit more cleaner. 
# A pseudo-code to start with always helps.

def make_scatter_circles():
    # sometimes too small circles may look like squares. No explicit reject coded in. 
    # only keeping the lower bound on the radius as 4 - this should reduce the count of bad data,
    # still overall training should be okay. 
    
    radius = torch.randint (4,14, (num_samples,))
    center = torch.randint (0,28, (2, num_samples))
    index = (center[0, :] < radius)

    # Lets clamp the center such that the circle can stay with-in the frame of 28*28.
    # avoiding lambda or map and using indexing operations. 
    center[0, index ] = radius[index]
    index =  (center[0, :] > 28-radius)
    center[0, index ] = 28-radius[index]

    index = (center[1, :] < radius)
    center[1, index ] = radius[index]
    index =  (center[1, :] > 28-radius)
    center[1, index ] = 28-radius[index]


    print (center)
    print (center.shape)
    

    angles = torch.rand((1,num_dots*num_samples)) * 2 * math.pi
    small_perturb = torch.rand((1,num_dots*num_samples)) 
    angles = angles.reshape(num_dots, num_samples)
    small_perturb = small_perturb.reshape(num_dots, num_samples)
    #TBD: add the small_perturb to the radius ( ciel (2(small_perturn - 0.5 ) + radius) might cut it )

    images = torch.zeros ( (28,28, num_samples)) 
    debug(images.shape)

    # radius shape is 1, num_samples.  angles is num_dots, num_samples
    # the boardcast of radius * torch.cos / torch.sin will work, trailing dims are matching.

    debug ( center[0,:].shape )

    points_x =  radius * torch.cos(angles) + center[0,:]
    points_y =  radius * torch.sin(angles) + center[1,:]
    px = points_x.long().clamp(0,27)
    py = points_y.long().clamp(0,27)

    # expand is like dragging the black dot on excel and expanding downwards ( note: memory allocation is prevented)
    sample_index = torch.arange(0,num_samples).unsqueeze(0).expand(num_dots, num_samples)
    

    images[px, py, sample_index] = 1

    # Displayes if verbosity is INFO   
    display_images(images) 

    #torch.conv2d expects N, channel, height, weidht! 
    # This is different from the convention used in the ANN code implemented in numpy earlier where 
    # inputs and outpus were (d,N) (1, N). Here its N, <features> for conv2d, so lets permute and return

    ret_images = torch.permute(images.unsqueeze(2), (3, 2, 0, 1))
    debug (ret_images.shape)

    label = torch.full((num_samples,), CIRCLE)
    
    return (ret_images,label)


 

class DataInput (Dataset):
    def __init__(self):
        d1 = make_scatter_square()
        d2 = make_scatter_circles()
 
        self.dataSet = ( torch.concatenate((d1[0], d2[0]), axis=0), torch.concatenate( (d1[1], d2[1]), axis=0))
        

    def __len__(self) -> int:
        return self.dataSet[1].shape[0]

    def __getitem__(self, index):
        debug ("Returning an image and a label:", self.dataSet[0][index], self.dataSet[1][index])
        return  self.dataSet[0][index], self.dataSet[1][index]

if __name__ == "__main__":
    # Testing code.
    set_verbosity(OUTPUT)
    set_seed(73)
    d=DataInput ()
    for i in range(0,len(d)):
        print (d[i][1].item() )

