%
% Image reconstruction for 3D acquisition with the k-t undersampling
% in the k-z direction (new version)
% New version: coil images are reconstructed separately.
% (reference: Zhao et al., (2012) IEEE TMI 31:1809-1820)
%
% kdata: k-space raw data (5-D) [kx,ky,kz,channel,time lag]
%        for time-lock acquisition: time lag is the number of 
%          time delays (e.g., cardiac phases) relative to the
%          trigger event(e.g., cardiac gating).
%        for real-time acquisition: time lag is same as the
%          number of time frames.
% kzmode: k-z table file orgnization mode:
%        = 1: real-time mode
%        = 2: time-lock mode
% kzfile: ky index table file
% Nz:   number of z-phase encodings in full sampling images
% Nnav: number of center k-space lines (navigators) for calculating
%       temporal subspace. These lines are sampled all time points,i.e.,
%       the center k-space are fully sampled
% Nz_recon: number of reconstructed pixels in z-phase encoding direction
%       if = []: same as Nz
% Vt:  prior basis vectors [M x L] added to navigator temporal subspace
%      M: number of time points; L: model order
%      if = []: No priors               
% tfrange: time frame range for analysis [start end]
%          if =[]: all time frames
% varargin{1}: order of ps model (L), default = 32
%              ONLY valid if Vt is empty
% varargin{2}: [1x2] beta and mu values (default = 1e3 and 1e-6) 
% magimg: reconstructed magnitude images [y,x,z,time] 
% varargout{1}: complex channel images [y,x,z,ch,time]
% ----------------------------------------------------------------------------------------------------
%
%                       Unversity of Illinois at Chicago
%
%  7/6/2021   Qingfei Luo     original
%  4/4/2023   Qingfei Luo     use all the central k-space data for extracting the temporal subspace
%
% ----------------------------------------------------------------------------------------------------

function [magimg,varargout] = kt_sparse_3d_recon_new(kdata, kzmode, kzfile, Nz, Nnav, Ny_recon, Nz_recon, Vt, tfrange, varargin)

beta = 1e3;  % accuracy parameter (alpha in the paper, default = 1e3)
mu   = 1e-6; %1e-3; % regularization parameter ("lambda" in the paper, default = 1e-6) (if = 0: basic PS)

[Nf,Ny,Nsam,Nchan,Nfr] = size(kdata); % Nsam is the number of sampled k-y lines in one
if isempty(Ny_recon)
    Ny_recon = Ny;
end

nfillky = Ny_recon - Ny;
if nfillky>0
    homodyne_ntran = 2;
    homodyne_niter = 4;
end
kdata = permute(kdata,[3,1,4,2,5]);

if isempty(Nz_recon)
    Nz_recon = Nz;
end
nfillkz = Nz_recon-Nz; % number of zero k-ys filled in the k-space 
kztable = load(kzfile);
kztable = kztable(1:Nsam*Nfr);

if ~isempty(tfrange)
    kdata=kdata(:,:,:,:,tfrange(1):tfrange(2));
    Nfr = tfrange(2)-tfrange(1)+1;
    kztable = kztable(Nsam*(tfrange(1)-1)+1:Nsam*tfrange(2));
end

r = 32; % PS model order
if ~isempty(varargin)
   r = varargin{1};
   if length(varargin)>1
       tmp = varargin{2};
       beta = tmp(1);
       mu = tmp(2);
   end
end
fprintf('Model order %d \n', r);

% loop ky to recon all the x-z images 
kspace_recon = single(zeros(Nz_recon,Nf,Nchan,Ny_recon,Nfr)); % kspace after kt
% Vt_mean = ones(Nfr,1)/sqrt(Nfr);
rsvd = r;
Vtmean = ones(Nfr,1)/sqrt(Nfr);
Vtmeanc = complex(Vtmean, Vtmean);
Vtmeanc = Vtmeanc/norm(Vtmeanc);

if ~isempty(Vt)
    if ~isempty(tfrange)
        Vt = Vt(tfrange(1):tfrange(2),:);
    end
    
    Vt = Vt-repmat(mean(Vt),[Nfr,1]);
    Vt = Vt./repmat(sqrt(sum(Vt.^2)),[Nfr,1]);
%     Vt = Vt-repmat(min(Vt),[Nfr,1]);
    Vt = Vt./repmat(sqrt(sum(Vt.^2)),[Nfr,1]); % normalization
    Vt = complex(Vt,Vt);
    
%     drift1 = (0:Nfr-1)';  % linear drift
%     drift1 = drift1-repmat(mean(drift1),[Nfr,1]);
%     drift1 = drift1./repmat(sqrt(sum(drift1.^2)),[Nfr,1]);
%     drift1 = complex(drift1,drift1);
%     Vt = [Vt, drift1];
%     Vt = [Vt, Vtmeanc,drift1];   
%     Vt = [Vtmean,Vt];

    r = size(Vt,2)+r;
end

% parfor yid = 1:Ny  % use parallel processing
for cid = 1:Nchan
    % extract tempporal subspace signal
    tmpnav = zeros(Nnav*Nf*Ny,Nfr);
    for yid = 1:Ny
        kxzdata = kdata(:,:,cid,yid,:);
        [~, tmp, ~] = reorg_kdata(1, kxzdata, kzmode, kztable, Nz, Nnav); % get navigator data for this y
        tmpnav(Nnav*Nf*(yid-1)+1:Nnav*Nf*yid,:) = tmp;
    end
    % determine the temporal subspace from navigator data
    [~, ~, Vt_r] = svds(tmpnav, rsvd);
    if isempty(Vt) % PCA (SVD)
        Vt_r = Vt_r';  % principle temporal signal components (temporal subspace basis)
    else
        Vt_r = [Vt_r, Vt];
        
        [tmp, ~] = qr(Vt_r); % Gram–Schmidt orthogonalization (equivalent to QR decomposition)
        Vt_r = tmp(:,1:r);

        Vt_r = Vt_r';
    end
        
    for yid = 1:Ny
        kxzdata = kdata(:,:,cid,yid,:);
        [kxzdata, Navdata, Mask] = reorg_kdata(1, kxzdata, kzmode, kztable, Nz, Nnav); % reorgnize kdata

        % determine the spatial subspace and reconstruct complex channel images
        Us_r0 = zeros(Nz*Nf, r);
        Us_r = ps_sparse_recon(kxzdata, Mask, Us_r0, Vt_r, mu, beta, 'xf_sparse', Nz, Nf, 1, Nfr);
        recon_pssp = Us_r*Vt_r;
        recon_pssp = reshape(recon_pssp,Nz,Nf,1,Nfr); 

        % tranform complex images to k-space
        for ti=1:Nfr
                fftpssp = fftshift(fft2(squeeze(recon_pssp(:,:,1,ti)))); % transform pssp image to k-space
                % zero-padding for partial k-z
                if nfillkz>0 % zill-filling k-space 
                    tmp = zeros(nfillkz/2,Nf,1,Nfr);
                    fftpssp = [tmp;fftpssp;tmp];
                    fftpssp = ifftshift(fftpssp);
                end
                kspace_recon(:,:,cid,yid,ti) = fftpssp;
        end
    end
end

% if nfillky>0  % apply homodyne to partial k-y
%     for zid = 1:Nz_recon
%         for ti=1:Nfr
%             for ci=1:Nchan
%                 tmp = squeeze(kspace_recon(zid,:,ci,:,ti));
%                 tmp = homodyne(tmp,Ny, homodyne_ntran, homodyne_niter, 'kxky', 'kxky');
%                 kspace_recon(zid,:,ci,:,ti) = tmp;
%             end
%         end
%     end
% end


% magnitude image
%     mag_fs = squeeze(sum(image_xyt.*conj(image_xyt),3)); % SOS combine
%     mag_pssp = squeeze(sum(recon_pssp.*conj(recon_pssp),3));
magimg = single(zeros(Ny_recon,Nf,Nz_recon,Nfr)); % channel-averaged magnitude images
compimg = single(zeros(Ny_recon,Nf,Nz_recon,Nchan,Nfr));
chimg = single(zeros(Ny_recon,Nf,Nz_recon,Nchan));
kspace_recon = permute(kspace_recon,[4,2,1,3,5]);
for k = 1:Nfr
    for ch=1:Nchan
        chimg(:,:,:,ch) = ifftshift(ifftn(squeeze(kspace_recon(:,:,:,ch,k))));
    end
    compimg(:,:,:,:,k) = chimg;
    magimg(:,:,:,k) = mean(sqrt(chimg.*conj(chimg)), 4); % chanel-averaged magnitude image
end
varargout{1} = compimg;


