import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch_dct as dct
from torch.autograd import grad

def plotGrid(X, Y, col='b'):
    n, k = X.shape
    for i in range(n):
        plt.plot(X[i,:], Y[i,:],col)
        plt.plot(X[:,i], Y[:,i],col)
    plt.show()
    return

def solvePoisson(f, kappa, h=0):
    # Solve (alpha*Delta + I)v = f
    dev = f.device
    Nx = f.shape[-1]
    Ny = f.shape[-2]
    Lx, Ly  = 2, 2 # Nx, Ny 
    dx, dy = Ly/Nx, Lx/Ny

    # modified wavenumber
    kx = torch.arange(Nx).to(dev)
    ky = torch.arange(Ny).to(dev)
    mwx = 2*(torch.cos(torch.pi*kx/Nx)-1)/dx**2
    mwy = 2*(torch.cos(torch.pi*ky/Ny)-1)/dy**2

    # 2D DCT of f (Right hand side)
    fhat = dct.dct_2d(f)
    #MWY, MWX  = torch.meshgrid(mwx,mwy)
    MWX, MWY  = torch.meshgrid(mwy,mwx)
    MXY = MWX+MWY
    MXY = MXY.unsqueeze(0).unsqueeze(0)
    MXY = MXY*kappa.reshape(1,-1,1,1)
    uhat = fhat/(-h*MXY + 1)
    #if uhat[:,:,0,0].norm()>1e4:
    #    uhat[:,:,0,0] = 0
    # Inverse 2D DCT
    u = dct.idct_2d(uhat)
    
    return u


def sub2ind(b, c, i, j, dims):
    idx = b*dims[1]*dims[2]*dims[3] + c*dims[2]*dims[3] + i*dims[3] + j
    
    return idx

class mass_preserving_advection(nn.Module):
    def __init__(self):
        super(mass_preserving_advection, self).__init__()

    def forward(self, input_image, U, V):
        
        device = input_image.device
        meshSize = (input_image.shape[2],input_image.shape[3])
        y, x = torch.meshgrid(torch.arange(meshSize[0]), torch.arange(meshSize[1]))
        x = x.to(device)
        y = y.to(device)
    
        # Get image dimensions
        B, C, H, W = input_image.shape
        
        # Calculate new coordinates using forward push
        X_new = x.unsqueeze(0) + U
        Y_new = y.unsqueeze(0) + V
        
        # Clamp coordinates
        X_new = X_new.clamp(0, W - 1)
        Y_new = Y_new.clamp(0, H - 1)
        
        # Compute floor and ceil indices
        x0 = X_new.floor().long()
        y0 = Y_new.floor().long()
        x1 = (x0 + 1).clamp(max=W - 1)
        y1 = (y0 + 1).clamp(max=H - 1)
        
        # Compute weights for bilinear interpolation
        w00 = (X_new - x0.float()) * (Y_new - y0.float())
        w01 = (X_new - x0.float()) * (y1.float() - Y_new)
        w10 = (x1.float() - X_new) * (Y_new - y0.float())
        w11 = (x1.float() - X_new) * (y1.float() - Y_new)
        
        b = torch.arange(B).reshape(-1, 1, 1, 1).to(device)
        c = torch.arange(C).reshape(1, -1, 1, 1).to(device)
        # Compute flat indices for scatter_add_
        dims = x0.shape
        
        # Target indices
        flat_indices00 = sub2ind(b, c, y0, x0, dims).reshape(-1)
        flat_indices01 = sub2ind(b, c, y1, x0, dims).reshape(-1)
        flat_indices10 = sub2ind(b, c, y0, x1, dims).reshape(-1)
        flat_indices11 = sub2ind(b, c, y1, x1, dims).reshape(-1)
        
        # Initialize output image
        flat_image = input_image.reshape(-1)
        output_image = torch.zeros_like(flat_image)
        
        output_image.index_add_(0, flat_indices00, flat_image*w00.reshape(-1)) 
        output_image.index_add_(0, flat_indices01, flat_image*w01.reshape(-1)) 
        output_image.index_add_(0, flat_indices10, flat_image*w10.reshape(-1)) 
        output_image.index_add_(0, flat_indices11, flat_image*w11.reshape(-1)) 

        # Reshape output image
        output_image = output_image.view(B, C, H, W) 
        return output_image


def CLP(dim_in, dim_out, shape=[256,256], kernel_size=[3,3]):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out,kernel_size=kernel_size, padding=kernel_size[0]//2),
        nn.LayerNorm(shape),
        nn.SiLU(),
        nn.Conv2d(dim_out, dim_out,kernel_size=kernel_size, padding=kernel_size[0]//2)
    )
    
class reactionBlock(nn.Module):
    def __init__(self, c, sz):
        
        super(reactionBlock, self).__init__()
        #self.sigma = nn.Parameter(0.1*torch.ones(1, c, sz[0], sz[1]))
        self.Conv1 = nn.Conv2d(c, c, kernel_size=[5,5], padding=2)
        self.LN1    = nn.LayerNorm(normalized_shape=sz)
        self.Conv2 = nn.Conv2d(c, c,kernel_size=[1,1])
        self.TimeEmbed  = nn.Parameter(1e-4*torch.randn(1,c,sz[0],sz[1]))
        self.TN         = CLP(c, c, sz)
        
    def forward(self, x, t):
        
        te = t.reshape([-1,1,1,1])*self.TimeEmbed
        te = self.TN(te) 
        x = self.Conv1(x + te)
        x = self.LN1(x)
        #x = F.silu(x)
        #x = self.sigma*x
        #x = self.LN2(x)
        x = F.silu(x)
        x = self.Conv2(x)
     
        return x

class advectionMassBlock(nn.Module):
    def __init__(self, c, mesh_size):
        super(advectionMassBlock, self).__init__()
        self.Adv = mass_preserving_advection()
        #self.Adv = color_preserving_advection()
        self.ConvU1 = nn.Conv2d(c, c, kernel_size=[3,3], padding=1)
        self.LNU1    = nn.LayerNorm(normalized_shape=mesh_size)
        self.ConvU2 = nn.Conv2d(c, c,kernel_size=[3,3], padding=1)
        self.TimeEmbedU  = nn.Parameter(1e-4*torch.randn(1,c,mesh_size[0],mesh_size[1]))
        self.TNU         = CLP(c, c,mesh_size)
        self.ConvV1 = nn.Conv2d(c, c, kernel_size=[3,3], padding=1)
        self.LNV1    = nn.LayerNorm(normalized_shape=mesh_size)
        self.ConvV2 = nn.Conv2d(c, c,kernel_size=[3,3], padding=1)
        self.TimeEmbedV  = nn.Parameter(1e-4*torch.randn(1,c,mesh_size[0],mesh_size[1]))
        self.TNV         = CLP(c, c,mesh_size)
        
    def forward(self, x, t):
        
        teU = t.reshape([-1,1,1,1])*self.TimeEmbedU
        teU = self.TNU(teU) 
        teV = t.reshape([-1,1,1,1])*self.TimeEmbedV
        teV = self.TNV(teV) 
        
        U = self.ConvU1(x + teU)
        U = self.LNU1(U)
        U = F.silu(U)
        U = self.ConvU2(U)
        
        V = self.ConvV1(x + teV)
        V = self.LNV1(V)
        V = F.silu(V)
        V = self.ConvV2(V)
        
        x = self.Adv(x, U, V)
        return x

class advectionColorBlock(nn.Module):
    def __init__(self, c,  mesh_size):
        super(advectionColorBlock, self).__init__()
        self.Adv = color_preserving_advection()
        #self.Adv = mass_preserving_advection()
        
        self.ConvU1 = nn.Conv2d(c, c, kernel_size=[3,3], padding=1)
        self.LNU1    = nn.LayerNorm(normalized_shape=mesh_size)
        # self.ConvU2 = nn.Conv2d(c, 1,kernel_size=[3,3], padding=1, bias=False)
        self.ConvU2 = nn.Conv2d(c, c,kernel_size=[3,3], padding=1, bias=False)
        self.ConvU2.weight = nn.Parameter(1e-4*torch.randn(c, c, 3, 3))

        self.TimeEmbedU  = nn.Parameter(1e-4*torch.randn(1,c,mesh_size[0],mesh_size[1]))
        self.TNU         = CLP(c, c,mesh_size)
        self.ConvV1 = nn.Conv2d(c, c, kernel_size=[3,3], padding=1)
        self.LNV1    = nn.LayerNorm(normalized_shape=mesh_size)
        #self.ConvV2 = nn.Conv2d(c, 1,kernel_size=[3,3], padding=1, bias=False)
        self.ConvV2 = nn.Conv2d(c, c,kernel_size=[3,3], padding=1, bias=False)
        self.ConvV2.weight = nn.Parameter(1e-4*torch.randn(c, c, 3, 3))

        self.TimeEmbedV  = nn.Parameter(1e-4*torch.randn(1,c,mesh_size[0],mesh_size[1]))
        self.TNV         = CLP(c, c,mesh_size)
        
    def forward(self, x, t=[]):
        
        if len(t)==0:
            t = torch.zeros(x.shape[0], device=x.device)

        nw, nh = x.shape[2], x.shape[3]
        teU = t.reshape([-1,1,1,1])*self.TimeEmbedU
        teU = self.TNU(teU) 
        teV = t.reshape([-1,1,1,1])*self.TimeEmbedV
        teV = self.TNV(teV) 
        
        U = self.ConvU1(x) + teU
        U = self.LNU1(U)
        U = F.silu(U)
        U = self.ConvU2(U)
        
        V = self.ConvV1(x) + teV
        V = self.LNV1(V)
        V = F.silu(V)
        V = self.ConvV2(V)
        U, V = U/nw, V/nw 
        #x = self.Adv(x, U, V)

        xr = x.reshape(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3])
        Ur = U.reshape(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3])
        Vr = V.reshape(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3])
        xr = self.Adv(xr, Ur, Vr)
        x  = xr.reshape(x.shape)
        

        return x


class color_preserving_advection(nn.Module):
    def __init__(self):
        super(color_preserving_advection, self).__init__()
        
    
    def forward(self, T, U, V):
        
        shape = (T.shape[2],T.shape[3])
        device = T.device
        grid_h, grid_w = shape[0], shape[1]
        y, x = torch.meshgrid(torch.linspace(-1, 1, grid_h), torch.linspace(-1, 1, grid_w))
        self.grid = torch.stack((x, y), dim=-1).unsqueeze(0).unsqueeze(0).to(device)

        UV = torch.stack((U, V), dim=-1)
        transformation_grid = self.grid + UV
        #Th = F.grid_sample(T, transformation_grid.squeeze(1))
        Th = F.grid_sample(T, transformation_grid.squeeze(1), align_corners=True)

        return Th


class diffusionBlock(nn.Module):
    def __init__(self, out_c):
        super(diffusionBlock, self).__init__()
        self.kappa = nn.Parameter(-2*6.9*torch.ones(1, out_c, 1))

    def forward(self, x, dt):  # compute
        kappa = torch.exp(self.kappa)  
        u = solvePoisson(x, kappa, h=dt)
        return u

class diffusion_reaction_net(nn.Module):
    def __init__(self, in_c, hid_c, out_c, nlayers=16, imsz=[256, 256]):
        super(diffusion_reaction_net, self).__init__()
        
        self.nlayers = nlayers
        #dropout_rate=0.05
        # Opening Layer
        self.Open = CLP(in_c, hid_c, imsz)
        # Main block
        #self.dropout = nn.Dropout(dropout_rate)
        self.Diffusion = diffusionBlock(hid_c)
        self.Reaction  = reactionBlock(hid_c, imsz)
        self.Advection = advectionColorBlock(hid_c, imsz)
        #self.Advection = advectionMassBlock(hid_c, imsz)
          
        
        # Close net
        #self.Close = nn.Conv2d(hid_c, out_c,kernel_size=1)
        self.Close = CLP(hid_c, out_c, imsz) #nn.Parameter(torch.randn(out_c, hid_c, 1, 1)*1e-2)
        self.h     = 1/imsz[0]
                
    def forward(self, x, t):
        
        z = self.Open(x)
        for i in range(self.nlayers):
            #print(i)
            # Advection
            z = self.Advection(z,t)
            # Reaction step
            dz = self.Reaction(z, t)
            z  = z + self.h*dz
            #z  = self.dropout(z)

            # Diffusion step
            z = self.Diffusion(z, self.h) 
            
        x = self.Close(z)
        #x = F.conv2d(z, self.Close)
        
        return x

class resnet(nn.Module):
    def __init__(self, cfg, in_c, hid_c, out_c, nlayers=16, imsz=[256, 256]):
        super(resnet, self).__init__()
        
        self.order = cfg["order"]
        self.integrator = cfg["integrator"]
        self.os = cfg["os"]

        self.nlayers = nlayers
        self.Open = CLP(in_c, hid_c, imsz)
        # Main block
        self.Adv = nn.ParameterList()
        self.DR  = nn.ParameterList()
        for i in range(nlayers):
            #Advi = advectionMassBlock(hid_c, imsz)
            Advi = advectionColorBlock(hid_c, imsz)
        
            DRi  = CLP(hid_c, hid_c, imsz, kernel_size=[5,5])
            self.Adv.append(Advi)
            self.DR.append(DRi)
        
        # Close net
        self.Close = nn.Parameter(torch.randn(out_c, hid_c, 1, 1)*1e-2) #CLP(hid_c, out_c, imsz) #
        self.h     = 1/imsz[0]
                
            
    def forward(self, x, t): 
        
        z = self.Open(x)
        if self.os == "lie":
            if self.order == "ADR":
                for i in range(self.nlayers):

                    z_adv = self.Adv[i](z,t)
                    if self.integrator == "FE":
                        # Diffusion Reaction step
                        dz = self.DR[i](z_adv)
                    elif self.integrator == "RK4":
                        # Compute the diffusion reaction step
                        k1 = self.DR[i](z_adv)
                        k2 = self.DR[i](z_adv + 0.5*self.h*k1)
                        k3 = self.DR[i](z_adv + 0.5*self.h*k2)
                        k4 = self.DR[i](z_adv + self.h*k3)
                        
                        dz = (k1 + 2*k2 + 2*k3 + k4)/6

                    z  = z_adv + self.h*dz    

            elif self.order == "DRA":
                for i in range(self.nlayers):
                    
                    if self.integrator == "FE":
                        # Diffusion Reaction step
                        dz = self.DR[i](z)
                    elif self.integrator == "RK4":
                        # Compute the diffusion reaction step
                        k1 = self.DR[i](z)
                        k2 = self.DR[i](z + 0.5*self.h*k1)
                        k3 = self.DR[i](z + 0.5*self.h*k2)
                        k4 = self.DR[i](z + self.h*k3)
                        
                        dz = (k1 + 2*k2 + 2*k3 + k4)/6

                    z  = z + self.h*dz
                    z = self.Adv[i](z,t)
        
        elif self.os == "strang":
            if self.order == "ADR":
                for i in range(self.nlayers):
                    z_adv = self.Adv[i](z,t/2)

                    if self.integrator == "FE":
                        # Diffusion Reaction step
                        dz = self.DR[i](z_adv)
                    elif self.integrator == "RK4":
                        # Compute the diffusion reaction step
                        k1 = self.DR[i](z_adv)
                        k2 = self.DR[i](z_adv + 0.5*self.h*k1)
                        k3 = self.DR[i](z_adv + 0.5*self.h*k2)
                        k4 = self.DR[i](z_adv + self.h*k3)
                        
                        dz = (k1 + 2*k2 + 2*k3 + k4)/6

                    z = z_adv + self.h*dz    
                    z = self.Adv[i](z,t/2)
                    

            elif self.order == "DRA":

                for i in range(self.nlayers):
                    if self.integrator == "FE":
                        # Diffusion Reaction step
                        dz = self.DR[i](z)
                        z  = z + 0.5*self.h*dz

                        z_adv = self.Adv[i](z,t)

                        dz = self.DR[i](z_adv)
                        z = z_adv + 0.5*self.h*dz


                    elif self.integrator == "RK4":
                        # Compute the diffusion reaction step
                        k1 = self.DR[i](z)
                        k2 = self.DR[i](z + 0.25*self.h*k1)
                        k3 = self.DR[i](z + 0.25*self.h*k2)
                        k4 = self.DR[i](z + 0.5*self.h*k3)
                        
                        dz = (k1 + 2*k2 + 2*k3 + k4)/6

                        z  = z + 0.5*self.h*dz
                        z_adv = self.Adv[i](z,t)

                        # Compute the diffusion reaction step
                        k1 = self.DR[i](z_adv)
                        k2 = self.DR[i](z_adv + 0.25*self.h*k1)
                        k3 = self.DR[i](z_adv + 0.25*self.h*k2)
                        k4 = self.DR[i](z_adv + 0.5*self.h*k3)
                        
                        dz = (k1 + 2*k2 + 2*k3 + k4)/6

                        z = z_adv + 0.5*self.h*dz

            
        x = F.conv2d(z, self.Close) #self.Close(z)
        
        return x

def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding


class UnetBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(UnetBlock, self).__init__()
        self.layerNorm = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

        #self.Transport = advectionColorBlock(out_c, [shape[1], shape[2]])

    def forward(self, x):
        out = self.layerNorm(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        #out = self.Transport(out)
        #out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out

class UNetAdv(nn.Module):
    def __init__(self, n_steps=10, time_emb_dim=100, arch=[1, 16, 32, 64, 128], dims=[64,64]):
        super(UNetAdv, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.DBlocks = nn.ModuleList()
        self.DConvs = nn.ModuleList()
        self.DTE = nn.ModuleList()
        
        # Down blocks
        for i in range(len(arch)-1):
            te = self._make_te(time_emb_dim, arch[i])
            blk = nn.Sequential(
                  UnetBlock((arch[i], dims[0], dims[1]), arch[i], arch[i+1]),
                  UnetBlock((arch[i+1], dims[0], dims[1]), arch[i+1], arch[i+1]),
                  UnetBlock((arch[i+1], dims[0], dims[1]), arch[i+1], arch[i+1]))
        
            down_cnv = nn.Sequential(nn.Conv2d(arch[i+1], arch[i+1], 4, 1, 1),
                                     nn.SiLU(),
                                     nn.Conv2d(arch[i+1], arch[i+1], 3, 2, 1))
            self.DBlocks.append(blk)
            self.DTE.append(te)
            self.DConvs.append(down_cnv)
            dims = [dims[0]//2, dims[1]//2]

        
        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, arch[-1])
        self.blk_mid = nn.Sequential(
            UnetBlock((arch[-1], dims[0], dims[1]), arch[-1], arch[-2]),
            UnetBlock((arch[-2], dims[0], dims[1]), arch[-2],arch[-2]),
            UnetBlock((arch[-2], dims[0], dims[1]), arch[-2], arch[-1])
        )

        self.UBlocks = nn.ModuleList()
        self.UConvs = nn.ModuleList()
        self.UTE = nn.ModuleList()
        # Up cycle
        for i in np.flip(range(len(arch)-1)):


            up = nn.Sequential(
                 nn.ConvTranspose2d(arch[i+1], arch[i+1], 4, 2, 1),
                 nn.SiLU(),
                 nn.ConvTranspose2d(arch[i+1], arch[i+1], 3, 1, 1))

            dims = [dims[0]*2, dims[1]*2]
            teu = self._make_te(time_emb_dim, arch[i+1]*2)
            if i != 0:
                blku = nn.Sequential(
                        UnetBlock((arch[i+1]*2, dims[0], dims[1]), arch[i+1]*2, arch[i+1]),
                        UnetBlock((arch[i+1], dims[0], dims[1]), arch[i+1], arch[i]),
                        UnetBlock((arch[i], dims[0], dims[1]), arch[i], arch[i]))
            else:
                blku = nn.Sequential(
                    UnetBlock((arch[i+1]*2, dims[0], dims[1]), arch[i+1]*2, arch[i+1]),
                    UnetBlock((arch[i+1], dims[0], dims[1]), arch[i+1], arch[i+1]),
                    UnetBlock((arch[i+1], dims[0], dims[1]), arch[i+1], arch[i+1]))
            
            self.UBlocks.append(blku)
            self.UTE.append(teu)
            self.UConvs.append(up)

        self.conv_out = nn.Conv2d(arch[1], arch[0], 3, 1, 1)
        self.conv_out.weight = nn.Parameter(torch.randn_like(self.conv_out.weight)*1e-4)

    def forward(self, x, t=[]):
        if len(t) == 0:
            t = torch.zeros(x.shape[0])
        t = self.time_embed(t.to(torch.int64))
        n = len(x)

        # down
        X = [x]
        for i in range(len(self.DBlocks)):

            te = self.DTE[i](t).reshape(n, -1, 1, 1)
            x  = self.DBlocks[i](x + te)
            X.append(x) 
            x  = self.DConvs[i](x)

        x = self.blk_mid(x + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

        cnt = -1
        for i in range(len(self.DBlocks)):
            x  = self.UConvs[i](x)
            x  = torch.cat((X[cnt],x), dim=1)  
            te = self.UTE[i](t).reshape(n, -1, 1, 1)
            x = self.UBlocks[i](x + te)  # 
            cnt = cnt-1
        
        out = self.conv_out(x)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

# net = diffusion_reaction_net(in_c=1, hid_c=64, out_c=1, nlayers=16, Mask=torch.ones(256, 256))
#net = UNetAdv(n_steps=10, time_emb_dim=100, arch=[10, 16, 32, 64, 128], dims=[64,64])

#U = torch.randn(7, 10, 64, 64)
#out = net(U)

#print('h')

