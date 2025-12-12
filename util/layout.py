import os
import glob
import rasterio
import rasterio.mask
import numpy as np
import matplotlib.pyplot as plt

class Layouts():
    def __init__(self):
        self.cmaps = {
            'ETa': 'jet_r',
            'ndvi': 'Greens',
        }

    def set_ticks(self, axe):
        yb,ye = axe.get_ylim()
        xb,xe = axe.get_xlim()
        y = ye - yb
        x = xe - xb
        y1,y2,y3 = yb + y*2/8, yb + y*4/8, yb + y*6/8
        x1,x2,x3 = xb + x*2/8, xb + x*4/8, xb + x*6/8
        axe.set_yticks([y1,y2,y3],[f"{y1:.2f}",f"{y2:.2f}",f"{y3:.2f}"])
        axe.set_xticks([x1,x2,x3],[f"{x1:.2f}",f"{x2:.2f}",f"{x3:.2f}"])

    def days_images_layout(self, tif_folder, exp_folder, field_shp, bbox, layout_name, data, d_min, d_max, colorbar):
        tif_paths = sorted(glob.glob(tif_folder+ "/*.tif"))
        spatial_extent = (bbox[0], bbox[2], bbox[1], bbox[3])
        # plot the images
        if len(tif_paths) > 3:
            rows = 2
            cols = -(-len(tif_paths) // rows)
            fig, axes = plt.subplots(rows, cols, figsize = (23.4,7.8), facecolor='white')
        else:
            rows = 1
            cols = len(tif_paths)
            fig, axes = plt.subplots(rows, cols, figsize = (23.4,7.8), facecolor='white')
        for idx, tif_path in enumerate(tif_paths):
            row = idx // cols
            col = idx % cols
            file_name = os.path.basename(tif_path).split('.')[0]
            with rasterio.open(tif_path) as src:
                tif_array = rasterio.mask.mask(src, field_shp.to_crs(src.crs).geometry, crop = True, nodata= np.nan)[0][0]
            if rows > 1:
                axes[row,col].set_title(file_name, fontweight= 'bold', pad = 10)
                axes[row,col].imshow(tif_array, cmap = self.cmaps[data], extent= spatial_extent, vmin = d_min, vmax = d_max)
                field_shp.boundary.plot(ax= axes[row,col], color= 'red',)
                self.set_ticks(axes[row,col])
            else:
                axes[col].set_title(file_name, fontweight= 'bold', pad = 10)
                axes[col].imshow(tif_array, cmap = self.cmaps[data], extent= spatial_extent, vmin = d_min, vmax = d_max)
                field_shp.boundary.plot(ax= axes[col], color= 'red',)
                self.set_ticks(axes[col])
        for i in range(len(tif_paths), cols*rows):
            axes[i//cols, i%cols].set_visible(False)
        # set colot bar
        if colorbar:
            plt.subplots_adjust(right= 1)
            cax = fig.add_axes([1, 0.1, 0.02, 0.8])
            sm = plt.cm.ScalarMappable(cmap= self.cmaps[data])
            sm.set_array([d_min, d_max])
            plt.colorbar(
                sm,cax=cax, shrink=0.75, aspect=50,
                )
            cax.set_ylabel(layout_name, fontsize= 16, labelpad = 10, fontweight= 'bold')

        os.makedirs(exp_folder, exist_ok= True)
        plt.savefig(os.path.join(exp_folder, f'{layout_name}.png'), bbox_inches = 'tight', pad_inches = 0.1)

    def season_layout(self, tif_folder, data, label, exp_folder, bbox, field_shp, layout_name, m_max, s_max):
        tif_paths_month = sorted(glob.glob(os.path.join(tif_folder, f'{data}_month')+ "/*.tif"))
        tif_path_season = sorted(glob.glob(os.path.join(tif_folder, f'{data}_season')+ "/*.tif"))[0]
        spatial_extent = (bbox[0], bbox[2], bbox[1], bbox[3])
        # plot months images
        if len(tif_paths_month) > 3:
            rows = 2
            cols = -(-len(tif_paths_month) // rows)
            fig, axes = plt.subplots(rows, cols+1, figsize = (23.4,7.8), layout= 'constrained', facecolor='white')
            gs = axes[0,-1].get_gridspec()
        else:
            rows = 1
            cols = len(tif_paths_month)
            fig, axes = plt.subplots(rows, cols+1, figsize = (23.4,7.8), layout= 'constrained', facecolor='white')
            gs = axes[-1].get_gridspec()
        for idx, tif_path in enumerate(tif_paths_month):
            row = idx // cols
            col = idx % cols
            file_name = os.path.basename(tif_path).split('.')[0]
            with rasterio.open(tif_path) as src:
                tif_array = rasterio.mask.mask(src, field_shp.to_crs(src.crs).geometry, crop = True, nodata= np.nan)[0][0]
            if rows > 1:
                axes[row,col].set_title(file_name, fontweight= 'bold', pad = 10)
                axes[row,col].imshow(tif_array, cmap = self.cmaps[data], vmin = 0, vmax = m_max, extent= spatial_extent)
                self.set_ticks(axes[row,col])
            else:
                axes[col].set_title(file_name, fontweight= 'bold', pad = 10)
                axes[col].imshow(tif_array, cmap = self.cmaps[data], vmin = 0, vmax = m_max, extent= spatial_extent)
                self.set_ticks(axes[col])
        for i in range(len(tif_paths_month), cols*rows):
            axes[i//cols, i%cols].set_visible(False)

        # plot month images color bar and marge the last column into one ax
        if len(tif_paths_month) > 3:
            sm = plt.cm.ScalarMappable(cmap= self.cmaps[data])
            sm.set_array([0,m_max])
            fig.colorbar(sm, ax= axes[-1,:-1], label= f"{label}/month", shrink = 0.6, pad= 0.1, orientation= 'horizontal', aspect= 50)
            for ax in axes[0:, -1]:
                ax.remove()
            ax_big = fig.add_subplot(gs[0:, -1])
        else:
            sm = plt.cm.ScalarMappable(cmap= self.cmaps[data])
            sm.set_array([0,m_max])
            fig.colorbar(sm,ax= axes[:-1], label= f"{label}/month", shrink = 0.6, pad= 0.1, orientation= 'horizontal', aspect= 50)
            axes[-1].remove()
            ax_big = fig.add_subplot(gs[-1])

        # plot season image
        file_name = os.path.basename(tif_path_season).split('.')[0]
        with rasterio.open(tif_path_season) as src:
            tif_array = rasterio.mask.mask(src, field_shp.to_crs(src.crs).geometry, crop = True, nodata= np.nan)[0][0]
        ax_big.imshow(tif_array, cmap = self.cmaps[data], vmin = 0, vmax = s_max, extent= spatial_extent)
        ax_big.set_title(file_name, fontweight= 'bold', pad = 10)
        self.set_ticks(ax_big)
        # season image color bar
        sm = plt.cm.ScalarMappable(cmap= self.cmaps[data])
        sm.set_array([0,s_max])
        fig.colorbar(sm,ax= ax_big, label= f"{label}/season", shrink = 0.8, pad= 0.1)

        # export layout
        os.makedirs(exp_folder, exist_ok= True)
        plt.savefig(os.path.join(exp_folder, f'{layout_name}.png'), bbox_inches = 'tight', pad_inches = 0.1)