import properties
import numpy as np
from .EM1DSimulation import set_mesh_1d
from SimPEG import Utils


class DataIO(properties.HasProperties):

    def __init__(self, **kwargs):
        super(DataIO, self).__init__(**kwargs)
        warnings.warn(
            "code under construction - API might change in the future"
        )

    def read_geosoft_xyz(self, data, header):
        pass

    def plot_profile_data(self, line_index=None):
        pass

    def plot_map_data(self, time_index=None, time=None):
        pass

    def plot_time_decay(self, time_index=None, time=None):
        pass


class ModelIO(properties.HasProperties):

    topography = properties.Array(
        "topography (x, y, z)", dtype=float,
        shape=('*', '*')
    )

    sigma = properties.Array(
        "Conductivity (S/m)", dtype=float
    )

    line = properties.Array(
        "Line", dtype=float, default=None
    )

    hz = properties.Array(
        "Vertical thickeness of 1D mesh", dtype=float
    )

    def __init__(self, **kwargs):
        super(ModelIO, self).__init__(**kwargs)
        warnings.warn(
            "code under construction - API might change in the future"
        )

    @property
    def unique_line(self):
        if getattr(self, '_unique_line', None) is None:
            if self.line is None:
                raise Exception("line information is required!")
            self._unique_line = np.unique(self.line)
        return self._unique_line

    @property
    def xyz_sigma(self):
        if getattr(self, '_xyz_sigma', None) is None:
            xyz_sigma = np.empty(
                (hz.size, self.topography.shape[0], 3), order='F'
            )
            for i_xy in range(self.topography.shape[0]):
                z = -mesh_1d.vectorCCx + self.topography[i_xy, 2]
                x = np.ones_like(z) * self.topography[i_xy, 0]
                y = np.ones_like(z) * self.topography[i_xy, 1]
                xyz_sigma[:, i_xy, :] = np.c_[x, y, z]
            self._xyz_sigma = xyz_sigma
        return self._xyz_sigma

    @property
    def mesh_1d(self):
        if getattr(self, '_mesh_1d', None) is None:
            if self.hz is None:
                raise Exception("hz information is required!")
            self._mesh_1d = set_mesh_1d(self.hz)
        return self._mesh_1d

    @property
    def Sigma(self):
        if getattr(self, '_Sigma', None) is None:
            if self.sigma is None:
                raise Exception("sigma information is required!")
            self._Sigma = self.sigma.reshape((hz.size, n_sounding), order='F')
        return self._Sigma

    def visualize_sigma_plan(
            self, i_layer=0, i_line=0, show_line=False,
            sigma=None, clim=None,
            ax=None, cmap='viridis', ncontour=20, scale='log',
            show_colorbar=True, aspect=1,
            contourOpts={}
    ):
        ind_line = self.line == self.unique_line[i_line]
        if sigma is not None:
            Sigma = np.exp(xc[i_iteration]).reshape(
                (hz.size, n_sounding), order='F'
            )
        else:
            Sigma = self.Sigma

        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = plt.subplot(111)

        if clim is None:
            vmin = np.percentile(Sigma, 5)
            vmax = np.percentile(Sigma, 95)
        else:
            vmin, vmax = clim

        if scale == 'log':
            contourOpts['vmin'] = np.log10(vmin)
            contourOpts['vmax'] = np.log10(vmax)
            norm = LogNorm()

        contourOpts['cmap'] = cmap

        im = Utils.plot2Ddata(
            self.topography[:, :2], Utils.mkvc(Sigma[i_layer, :]), scale=scale,
            ncontour=ncontour, ax=ax,
            contourOpts=contourOpts, dataloc=False,
        )

        out = ax.scatter(
            self.topography[:, 0], self.topography[:, 1],
            c=Sigma[i_layer, :], s=0.5, vmin=vmin, vmax=vmax,
            cmap=cmap, alpha=1, norm=norm
        )

        if show_line:
            ax.plot(
                self.topography[ind_line, 0], self.topography[ind_line, 0],
                'k.', ms=1
            )

        if show_colorbar:
            from mpl_toolkits import axes_grid1
            divider = axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = plt.colorbar(out, cax=cax)
            cb.set_label("Conductivity (S/m)")
        ax.set_aspect(aspect)
        ax.set_title(
            ("Conductivity at %.1f m below surface") % (self.mesh_1d.vectorCCx[i_layer])
        )
        ax.set_xlabel("Easting (m)")
        ax.set_xlabel("Northing (m)")
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        return out, ax

    def visualize_sigma_section(
        self, i_layer=0, i_line=0, line_direction='x',
        show_layer=False,
        sigma=None, clim=None, view_type='map',
        ax=None, cmap='viridis', ncontour=20, scale='log',
        show_colorbar=True, aspect=1,
        contourOpts={}
    ):
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ind_line = self.line == self.unique_line[i_line]
        if sigma is not None:
            Sigma = np.exp(xc[i_iteration]).reshape(
                (hz.size, n_sounding), order='F'
            )
        else:
            Sigma = self.Sigma

        if line_direction.lower() == 'x':
            yz = self.xyz_sigma[:, ind_line, :][:, :, [1, 2]].reshape(
                (int(self.hz.size*ind_line.sum()), 2), order='F'
            )

        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = plt.subplot(111)

        if clim is None:
            vmin = np.percentile(Sigma, 5)
            vmax = np.percentile(Sigma, 95)
        else:
            vmin, vmax = clim

        if scale == 'log':
            contourOpts['vmin'] = np.log10(vmin)
            contourOpts['vmax'] = np.log10(vmax)
            norm = LogNorm()

        contourOpts['cmap'] = cmap
        im = Utils.plot2Ddata(
            yz, Utils.mkvc(Sigma[:, ind_line]), scale='log', ncontour=40,
            dataloc=False, ax=ax,
            contourOpts=contourOpts
        )
        ax.fill_between(
            self.topography[ind_line, 1], self.topography[ind_line, 2],
            y2=yz[:, 1].max(), color='w'
        )

        out = ax.scatter(
            self.topography[ind_line, 0], self.topography[ind_line, 1],
            c=Utils.mkvc(Sigma[:, ind_line]), s=0.5, vmin=vmin, vmax=vmax,
            cmap=cmap, alpha=1, norm=norm
        )

        if show_layer:
            ax.plot(
                self.topography[ind_line, 1],
                self.topography[ind_line]-mesh_1d.vectorCCx[i_layer],
                '--', lw=1, color='grey'
            )

        if show_colorbar:
            from mpl_toolkits import axes_grid1
            divider = axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = plt.colorbar(out, cax=cax)
            cb.set_label("Conductivity (S/m)")

        ax.set_aspect(aspect)

        ax.plot(self.topography[ind_line, 1], self.topography[ind_line], 'kv')

        ax.set_ylim(800, 1020)
        ax.set_xlabel('Northing (m)')
        ax.set_ylabel('Elevation (m)')
        plt.gca().set_aspect(20)
        plt.tight_layout()
        pass
