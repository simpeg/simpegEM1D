from SimPEG.Problem import BaseProblem
from SimPEG import Maps
import properties
import scipy.sparse as sp
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from multiprocess import Pool

class AEMInterpolation(BaseProblem):

    _jtj_diag = None
    
    locations = properties.Array(
        "(x,y,z) locations where AEM soundings are defined",
        required=True,
        shape=('*', 3),  
        dtype=float  # data are floats
    )

    hz_aem = properties.Array(
        "(x,y,z) locations where AEM soundings are defined",
        required=True,
        shape=('*',),  
        dtype=float  # data are floats
    )
    
    n_cpu = properties.Integer(
        "Number of CPU",
        default=2,
        required=True
    )
    
    def __init__(self, mesh, **kwargs):
        BaseProblem.__init__(self, mesh, **kwargs)
        self.modelMap = kwargs.pop('modelMap', Maps.IdentityMap(mesh))
    
    @property
    def modelMap(self):
        "A SimPEG.Map instance."
        return getattr(self, '_modelMap', None)


    @modelMap.setter
    def modelMap(self, val):
        val._assertMatchesPair(self.mapPair)
        self._modelMap = val
    
    @property
    def n_layer(self):
        return self.hz_aem.size

    @property
    def n_sounding(self):
        return self.locations.shape[0]

    @property
    def xyz(self):
        if getattr(self, '_xyz', None) is None:
            z_center_aem = (np.r_[0, np.cumsum(self.hz_aem)[:-1]] + np.cumsum(self.hz_aem)) * 0.5
            x = np.repeat(self.locations[:,0], self.hz_aem.size)
            y = np.repeat(self.locations[:,1], self.hz_aem.size)
            z =  np.repeat(self.locations[:,2], self.hz_aem.size) - np.tile(z_center_aem, (self.n_sounding, 1)).flatten()            
            self._xyz = np.c_[x, y, z]
        return self._xyz
            
    @property
    def G(self):
        if getattr(self, '_G', None) is None:
            print (">> Computing G matrix")
            pool = Pool(self.n_cpu)
            self._G = pool.map(
                get_projection_matrix, 
                [(self.locations[ii,:], self.hz_aem, self.mesh.hx, self.mesh.hy, self.mesh.hz, self.mesh.x0) for ii in range(self.locations.shape[0])]
            )
            pool.close()
            pool.join()
            self._G = sp.vstack(self._G)   
            print (">> Finished computing G matrix")
        return self._G
    
    def fields(self, m):
        return self.G.dot(self.modelMap * m)

    def getJ(self, m, f=None):
        """
            Sensitivity matrix
        """

        if self.modelMap is not None:
            dmudm = self.modelMap.deriv(m)
            return self.G*dmudm
        else:
            return self.G

    def Jvec(self, m, v, f=None):
        return self.G.dot(self.modelMap.deriv(m) * v)

    def Jtvec(self, m, v, f=None):
        return self.modelMap.deriv(m).T*self.G.T.dot(v)
    
    def getJtJdiag(self, m, W=None):
        if self._jtj_diag is not None:
            return self._jtj_diag
        else:
            # Note: there is a memory peak
            J_matrix = W*self.getJ(m)
            I, J = J_matrix.nonzero()
            values = J_matrix.data
            J2 = sp.coo_matrix((values**2, (I, J)), shape=J_matrix.shape).tocsr()
            self._jtj_diag = J2.T * np.ones(J2.shape[0])
            threshold = np.percentile(self._jtj_diag[self._jtj_diag>0.], 5)
            self._jtj_diag += threshold
            return self._jtj_diag
        
    def get_nearest_interpolation_3d(self, sigma_1d):
        f_int = NearestNDInterpolator(self.xyz, sigma_1d)
        return f_int(self.mesh.gridCC)
        
def get_projection_matrix(args):
    import numpy as np
    import scipy.sparse as sp
    from SimPEG import Mesh, Utils

    location, hz_aem, hx, hy, hz, x0 = args
    
    def get_circle(center_points, r, n):
        theta = np.linspace(-np.pi, np.pi, n+1)
        x = r*np.cos(theta) + center_points[0]
        y = r*np.sin(theta) + center_points[1]
        return np.c_[x,y]    
    
    def in_hull(p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        from scipy.spatial import Delaunay
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0 
    
    def get_1d_volumetric_average_matrix(z_loc, hz_aem, z):
        """
            Compute weights for volumetric average

            Parameters
            ----------

            hz_aem: (n_layer,) array_like
                Thickness of the AEM layers
            z: array_like
                Nodal grid of 1D mesh
            Returns
            -------
            P_volume: (n_layer, z.size), sparse matrix
                averaging matrix

        """
        z_top = z[1:]
        z_bottom = z[:-1]
        center = (z_top + z_bottom)*0.5
        zmin = z_top.min()
        zmax = z_bottom.max()
        depth = z_loc - np.r_[0., np.cumsum(hz_aem)][:]
        z_aem_top = depth[:-1]
        z_aem_bottom = depth[1:]

        # assume lithology log always start with zero depth
        n_layer = hz_aem.size
        values = []
        J = []
        I = []

        for i_layer in range(n_layer):
            top = z_aem_top[i_layer]
            bottom = z_aem_bottom[i_layer]
            inds_in = np.argwhere(np.logical_and(z<=top, z>=bottom)).flatten()
            dx_aem = top-bottom
            if inds_in.sum() != 0:
                z_in = z[inds_in] 
                dx_in = np.diff(z_in)    
                # When both top and bottom are exact
                if np.logical_and(z_in[-1] == top, z_in[0] == bottom):
                    inds = inds_in.copy()
                    dx = dx_aem
                # When top is exact
                elif np.logical_and(z_in[-1] == top, z_in[0] != bottom):
                    inds_bottom = inds_in[0]-1
                    inds = np.r_[inds_bottom, inds_in]
                    dx_bottom = z[inds_in[0]] - bottom
                    dx = np.r_[dx_bottom, dx_in]
                # When bottom is exact
                elif np.logical_and(z_in[-1] != top, z_in[0] == bottom):
                    inds_top = inds_in[-1]+1
                    inds = np.r_[inds_in, inds_top]
                    dx_top = top - z[inds_in[-1]]
                    dx = np.r_[dx_in, dx_top]
                # Others
                else:
                    inds_bottom = inds_in[0]-1
                    inds_top = inds_in[-1]+1
                    dx_bottom = z[inds_in[0]] - bottom
                    dx_top = top - z[inds_in[-1]]
                    inds = np.r_[inds_bottom, inds_in, inds_top]
                    dx = np.r_[dx_bottom, dx_in, dx_top]
                inds_center = inds[:-1]
            else:
                inds_center = np.argmin(abs(bottom-z))
                dx = dx_aem

            dx /= dx.sum()
            values.append(dx)
            J.append(inds_center)
            I.append(np.ones(inds_center.size, dtype=int) * i_layer)
        J = np.hstack(J)
        I = np.hstack(I) 
        values = np.hstack(values)
        P_volume = sp.coo_matrix((values, (I, J)), shape=(n_layer, center.size)).tocsr()
        return P_volume

    mesh_3d = Mesh.TensorMesh([hx, hy, hz], x0)

    z_center_aem = -(np.r_[0, np.cumsum(hz_aem)[:-1]] + np.cumsum(hz_aem)) * 0.5 + location[2]
    xy_top = get_circle(location[:2], hx.min(), 20)
    xy_bottom = get_circle(location[:2], np.sum(hz)*np.sqrt(3), 20)
    depth_top = location[2]
    depth_bottom = location[2]-np.sum(hz)
    xyz_top = np.c_[xy_top, depth_top*np.ones(xy_top.shape[0])]
    xyz_bottom = np.c_[xy_bottom, depth_bottom*np.ones(xy_top.shape[0])]
    xyz_cone = np.vstack((xyz_top, xyz_bottom))

    inds = in_hull(mesh_3d.gridCC, xyz_cone)

    nC = mesh_3d.nC
    n_cone = inds.sum()
    nCz = mesh_3d.nCz
    nCy = mesh_3d.nCy
    nCx = mesh_3d.nCx    

    I = np.argwhere(inds).flatten()
    J = np.argwhere(inds).flatten()
    weight_sum = inds.reshape((np.prod(nCx*nCy), nCz), order='F').sum(axis=0)
    values = np.repeat(1./weight_sum, np.prod(nCx*nCy))[I]
    P_cone = sp.coo_matrix(
        (values, (I, J)), shape=(nC, nC)
    ).tocsr() 
    I = np.repeat(np.arange(nCz), np.prod(nCx*nCy))
    J = np.arange(nC)
    P_avg = sp.coo_matrix(
        (np.ones(mesh_3d.nC), (I, J)), shape=(nCz, nC)
    ).tocsr()
    mesh_1d = Mesh.TensorMesh([mesh_3d.hz], x0=[mesh_3d.x0[2]])
    P_1D = get_1d_volumetric_average_matrix(location[2], hz_aem, mesh_1d.vectorNx)
    P = P_1D * P_avg * P_cone
    return P