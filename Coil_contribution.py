import numpy as np

class Coil_contribution:
    def __init__(self, filename=None):
        '''
        Initialization of a class which captures all operations related to the coil field on the computational domain
        Omega. The given file with a set of coils is read in the initialization of this class.
        :param filename: Name of the (text)file where the list of coils is stored.
        '''
        self.mu0 = 4 * np.pi * 1e-7
        self.ncoils=0
        if ( filename ):
            self.read_coils_file(filename)
            
    def read_coils_file(self, filename):
        '''
        Reads the list of coils with their position and current and stores this into an array.
        :param filename: Name of the (text)file where the list of coils is stored.
        :return: Array with size (the number of coils) x 3. Every row has 3 columns: the R-component of the coil
        position, the Z-component of the coil position and the current.
        '''

                
        file_coils = open(filename, "r") #"coils_symmetric.txt"
        content = np.array(file_coils.readlines())
        file_coils.close()

        coil_index_set = range(3, content.size)  # start at 3: ignore header and R=0 coils
        self.ncoils = len(coil_index_set)
        self.coil_arr = np.zeros((self.ncoils, 3))  # columns: R, Z, current
        

        for i in range(0, self.ncoils):
            coil = content[coil_index_set[i]].split()
            self.coil_arr[i, :] = float(coil[0]), float(coil[1]), float(coil[-1])

    def eval_single_coil_field(self, r,z, rc,zc,jc):
        from GS_kernels import G
        return G(r, z,rc,zc) * (self.mu0 *jc)
    
    def eval_multi_coil_field(self, r,z, r_coils,z_coils,j_coils):
        psi_cf=0.*r
        for rc,zc,jc in zip(r_coils, z_coils,j_coils):
            psi_cf+= self.eval_single_coil_field(r, z,rc,zc,jc)
        return psi_cf
    
    def eval_coil_field(self, r, z):
        '''
        Evaluate the coil field at position (r,z).
        :param r: R-component of the position where to compute the field.
        :param z: Z-component of the position where to compute the field.
        :return: The scalar psi of the coil field at position (r,z).
        '''
        return self.eval_multi_coil_field(r, z,self.coil_arr[:,0],self.coil_arr[:, 1],self.coil_arr[:, 2])


    def eval_dr_single_coil_field(self, r,z, rc,zc,jc):
        from GS_kernels import dG_dr,dG_dz
        return dG_dr(r, z,rc,zc) * (self.mu0 *jc)
    
    def eval_dz_single_coil_field(self, r,z, rc,zc,jc):
        from GS_kernels import dG_dr,dG_dz
        return dG_dz(r, z,rc,zc) * (self.mu0 *jc)

    def eval_grad_single_coil_field(self, r,z, rc,zc,jc):
        from GS_kernels import G
        return self.eval_dr_single_coil_field(r,z, rc,zc,jc),self.eval_dz_single_coil_field(r,z, rc,zc,jc)
    
    def eval_grad_multi_coil_field(self, r,z, r_coils,z_coils,j_coils):
        dpsi_dr_cf=0.*r
        for rc,zc,jc in zip(r_coils, z_coils,j_coils):
            dpsi_dr_cf += self.eval_dr_single_coil_field(r, z,rc,zc,jc)
            
        dpsi_dz_cf=0.*r
        for rc,zc,jc in zip(r_coils, z_coils,j_coils):
            dpsi_dz_cf += self.eval_dz_single_coil_field(r, z,rc,zc,jc)
            
        return dpsi_dr_cf,dpsi_dz_cf
    
    def eval_grad_coil_field(self, r, z):
        '''
        Evaluate the coil field at position (r,z).
        :param r: R-component of the position where to compute the field.
        :param z: Z-component of the position where to compute the field.
        :return: The gradient of psi of the coil field at position (r,z).
        '''
        return self.eval_grad_multi_coil_field(r, z,self.coil_arr[:,0],self.coil_arr[:, 1],self.coil_arr[:, 2])
   


