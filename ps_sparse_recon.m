function Us_r = ps_sparse_recon(d, Mask, Us_r, Vt_r, mu, beta, regularizer, Np, Nf, Nchan, Nfr, varargin)
%
% This Matlab code performs image reconstruction from highly undersampled data using
% joint partial separability (PS) and sparsity constraints
%
% Inputs:
% d: measured data (1-D)
% Mask: k-t space sampling mask
% Us_r: initial value of Us_r
% Vt_r: basis for the temporal subspace [Nr x Nt]
% mu: regularization parameter
% beta: initial value of continuation parameter
% regularizer: spa_reg for spatial sparsity regularization
% Np: number of phase encodings
% Nf: number of frequence encodings
% Nchan: number of receiver channels
% Nfr: number of temporal frames
% varargin: tolorance parameters [1x2] for the pcg function and
%           change of Us_r
%           default = [1e-5, 1e-4]
% 
% Output:
% Us_r: basis for the spatial subspace
% 
% Bo Zhao, 11/27/2011, zhaobouiuc@gmail.com

% Reference: B. Zhao, J. P. Haldar, A. G. Christodoulou, and Z.-P. Liang, "Image Reconstruction 
% from Highly Undersampled (k, t)-Space Data with Joint Partial Separability
% and Sparsity Constraints", IEEE Trans. Medical Imaging, vol 31, pp. 1809-1820,  2012. 
% --------------------------------------------------------------------------------------------------
%
% Qingfei Luo   6/23/2020   allow for multiple channels (coils)
% Qingfei Luo   9/14/2023   allow for setting the tolerance parameters
%
% --------------------------------------------------------------------------------------------------

maxit = 200; % max iterations, default = 200
noutloop = 5;   % number of outerloops, default = 5
ninloop = 100;  % number of inner loops, default = 100

tol = 1e-5;  % tolerance used in the pcg function, default = 1e-5
tol_Us = 1e-4;  % tolerance of change of Us, default = 1e-4
if ~isempty(varargin)
    tmp = varargin{1};
    tol = tmp(1);
    tol_Us = tmp(2);
end 

r = size(Us_r, 2);
[Rind, Cind] = find(Mask);
bb = sparse(Rind, Cind, d, Np*Nf*Nchan, Nfr)*Vt_r';

Uk_r = 1/sqrt(Np*Nf*Nchan)*reshape(fft2(reshape(Us_r, Np, Nf, Nchan, r)), Np*Nf*Nchan, r);
Us_r_last = Us_r;
Vf_r = 1/sqrt(Nfr)*fft(Vt_r, [], 2);
% Vf_r = 1/sqrt(Nfr)*fftshift(fft(Vt_r, [], 2),2);

switch regularizer
    case 'xf_sparse'
        for mm = 1:noutloop
            fprintf('The outerloop is %d\n', mm);
            for nn = 1:ninloop
                fprintf('The interloop is %d\n', nn);
                Gs = sign(Us_r*Vf_r).*wthresh(abs(Us_r*Vf_r), 's', 1/beta)*Vf_r';
                Gk =  1/sqrt(Np*Nf*Nchan)*fft2(reshape(Gs, Np, Nf, Nchan, r));
                b  = bb + mu*beta/2*reshape(Gk, Np*Nf*Nchan, r);
                [Uk_r, ~, ~, ~, ~] = pcg(@(x) (normal_oper(x, Vt_r, Rind, Cind, Mask) + mu*beta/2*x), b(:), tol, maxit, [], [], Uk_r(:));
                Uk_r = reshape(Uk_r, Np*Nf*Nchan, r);
                Us_r = sqrt(Np*Nf*Nchan)*ifft2(reshape(Uk_r, Np, Nf,Nchan, r));
                Us_r = reshape(Us_r, Np*Nf*Nchan, r);
                relchange_Us = norm(Us_r(:) - Us_r_last(:))/norm(Us_r_last(:))
                Us_r_last = Us_r; 
                if(relchange_Us < tol_Us)
                    break;
                end  
            end
            beta = 10*beta;
        end
end
