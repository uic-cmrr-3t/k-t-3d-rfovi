%
%  Create k-y (phase encoding) index table for the k-t sparse acquisition         
%
%  To achieve the k-t undersampling, the central k-y lines (navigators) 
%  are fully sampled/acquired, and the outer  k-y lines are randomly 
%  undersampled at a time frame. The navigator signal is used 
%  to obain the basis of temporal space using SVD (i.e.,PCA),
%  and the spatial images are reconstructed using the PS-Sparse method
%  (Zhao et al., (2012) IEEE TMI 31:1809-1820).
% 
% acqmode: choice of aquisition mode
%
%           = 1: Conventional gradient/spin echo acquisition with
%                fully random k-y sampling in two scenarios:
%                (A) continous time frame acquisition (real-time)
%                (B) event time-locked acquisition
%                A: one whole image frame is acquired at one time
%                   point and then next whole image frame.
%                B: one k-y line is acquired for one image
%                  frame, and then one k-y line for next frame.
%                  The 2nd k-y line is acquired after the first
%                  k-y lines of all image frames are finished. The 
%                  acquisition is usually time locked to event markers
%                  in scenario B such as cardiac gatings.
%
%           = 2: Single-shot EPI acquisition with semi-random sampling
%                and real-time acquisition (same as mode 1A):
%                the same number of k-y lines are randomly
%                acquired in the upper and lower half k-space.
%               
%               
%           = 3: EPI-SPEEDI with limited random sampling:
%                The MR signal is acquired at different time
%                lags/phases, which are time locked to event
%                markers (e.g., cardiac gatings). Multiple echos are 
%                acquried at each time lag and each echo is used to
%                form a separate image. At each time lag, the k-y random 
%                sampling across echos is limited to a small ky range 
%                relative to the k-y of the first echo, e.g., 
%                between ky0-8 and ky0+8, where ky0 is the k-y location 
%                at the first echo. The sampling at different time lags 
%                is completely randomized in the full k-y range.            
%           
%
% Np: number of phase encodings (k_y) in the full sampling image
%
% Nlag: when acqmode = 3: number of time lags relative to event markers, 
%                    e.g., cardiac phases relative to the cardiac gating
%       when acqmode = other: numbter of image time frames (time points)
%
% Necho: when acqmode = 3: number of echos at each time lag
%        when acqmode = other: ignored (always 1)
% Nsam: total number of acquired k-y lines in the undersampling image
%       MUST be a even number
% Nnav: number of sampled k-y lines in the central k-sapce (navigators).
%       navigator lines are fully sampled at each time frame
%       MUST be a even number
%      
% varargin{1}: parameters related to acquistion mode
%              acqmode = 1: k-y acquisition order in each image frame, 
%                           =1 (default): all ascending
%                           =2: all random
%                           =3: fixed distributed navigators and other random
%              acqmode = 2: max ky step change allowed:
%                           default: (Np-Nnav)/2                          
%              acqmode = 3: k-y shift range (default = [-1 1]) 
% kytable: k-y index table (phase order table) used in image acquisition
%        when acqmode = 1: [Nsam, Nlag]
%        when acqmode = 2: [Nsam, Nlag]
%        when acqmode = 3: [Nsam, Necho, Nlag]
% varargout{1}: navigator index [Nnav, 1]
%
% ------------------------------------------------------------------------------------------
% 
% 7/22/2020     Qingfei Luo     original 
%
% ------------------------------------------------------------------------------------------

function [kytable,varargout] = create_kytable(acqmode, Np, Nlag, Necho, Nsam, Nnav, varargin)

nav_ind = (Np/2-Nnav/2+1:Np/2+Nnav/2)'; % # navigator k-y indices
switch acqmode
    case 1  % conventional gradient/spin echo
        if ~isempty(Necho)
            if Necho>1
                fprintf('number of echos ignored. it is always 1. \n');
            end
        end
        Nfr = Nlag; % total number of time frames (time points)
        kytable = zeros(Nsam, Nfr); % k_y acquisition sequence
        out_ind = [1:Np/2-Nnav/2,Np/2+Nnav/2+1:Np]; % out k-y indices
        Nsamout = Nsam - Nnav; % number of sampled k-y in outer k-space per time frame
        Npout = Np-Nnav; % number of k-y in outer k-space
        kytable(1:Nnav,:) = repmat(nav_ind,1,Nfr);
        for ti = 1:Nfr  % randomly sample out k-y locations in each frame
            tmp = randperm(Npout);
            tmpind = out_ind(tmp(1:Nsamout)); % selected out k-y
            kytable(Nnav+1:end,ti)=tmpind;
        end
        if isempty(varargin)
            sortmode = 1;
        else
            sortmode = varargin{1};
        end
        switch sortmode % sort k-y order in each image frame
            case 1  % ascending
                kytable = sort(kytable,1);
            case 2  % random
                for ti = 1:Nfr
                    tmp = randperm(Nsam);
                    kytable(:,ti)=kytable(tmp,ti);
                end
            case 3 % fixed navigator and random outer ky
                tmpstep = ceil(Nsam/Nnav);
                tmpnavord = 1:tmpstep:Nsam;
                tmpkytable = kytable;
                for ti = 1:Nfr
                    tmp = randperm(Nsamout);
                    kytable(Nnav+1:end,ti)=kytable(Nnav+tmp,ti);
                    tmpoff = Nnav;
                    for j=1:Nnav-1
                        tmpkytable(tmpnavord(j),ti) = kytable(j,ti);
                        tmpkytable(tmpnavord(j)+1:tmpnavord(j+1)-1,ti) = kytable(tmpoff+1:tmpoff+tmpstep-1,ti);
                        tmpoff = tmpoff+tmpstep-1;
                    end
                    tmpkytable(tmpnavord(Nnav),ti) = kytable(Nnav,ti);
                    tmpkytable(tmpnavord(Nnav)+1:end,ti) = kytable(tmpoff+1:end,ti);
                end
                kytable = tmpkytable;
        end
        
    case 2 % single-shot EPI
        Nfr = Nlag; % total number of time frames (time points)
        kytable = zeros(Nsam, Nfr); % k_y acquisition sequence
        out_ind1 = (1:Np/2-Nnav/2); % out lower k-y indices
        out_ind2 = (Np/2+Nnav/2+1:Np); % out upper k-y indices
        Nsamout = Nsam - Nnav; % number of sampled k-y in outer k-space per time frame
        Npout = Np-Nnav; % number of k-y in outer k-space
        maxkystep = (Np-Nnav)/2;
        if ~isempty(varargin)
            maxkystep = varargin{1};
            if maxkystep > (Np-Nnav)/2
                maxkystep = (Np-Nnav)/2;
            end
        end
        for ti = 1:Nfr  % randomly sample out k-y locations in each frame
            tmp = randperm(Npout/2);  % select lower outer k-y
            tmpind = out_ind1(tmp(1:Nsamout/2));
            if maxkystep > 0 % check if ky step change is too large
                tmpind = sort(tmpind);
                kystep = diff(tmpind);
                largeid = find(kystep > maxkystep);
                if ~isempty(largeid) % reduce ky step changes
                    for jj=1:length(largeid)
                        tmpind(largeid(jj)+1:end) = tmpind(largeid(jj)+1:end)+maxkystep-kystep(largeid(jj));
                    end
                end
            end
            kytable(1:Nsamout/2,ti)=tmpind;
            kytable(Nsamout/2+1:Nsamout/2+Nnav,ti) = nav_ind; % fill navigator k-y
            tmp = randperm(Npout/2);  % select upper outer k-y
            tmpind = out_ind2(tmp(1:Nsamout/2)); 
            if maxkystep > 0 % check if ky step change is too large
                tmpind = sort(tmpind);
                kystep = diff(tmpind);
                largeid = find(kystep > maxkystep);
                if ~isempty(largeid) % reduce ky step changes
                    for jj=1:length(largeid)
                        tmpind(largeid(jj)+1:end) = tmpind(largeid(jj)+1:end)+maxkystep-kystep(largeid(jj));
                    end
                end
            end
            kytable(Nsamout/2+Nnav+1:Nsam,ti)=tmpind;
        end
        kytable = sort(kytable,1);
        
    case 3    % EPI-SPEEDI (limited random undersampling)
        Nfr = Nlag*Necho; % number of time frames (time points)
        kytable = zeros(Nsam, Nfr); % k_y acquisition sequence
        out_ind = [1:Np/2-Nnav/2,Np/2+Nnav/2+1:Np]; % out k-y indices
        Nsamout = Nsam - Nnav; % number of sampled k-y in outer k-space per time frame
        kytable(1:Nnav,:) = repmat(nav_ind,1,Nfr);
        Npout = Np-Nnav; % number of k-y in outer k-space
        kytable = reshape(kytable,Nsam,Necho,Nlag);
        if isempty(varargin)
            kyr = [-1,1];  % range of ky relative to 1st echo
        else
            kyr = varargin{1};
        end
        for ii = 1:Nlag  % loop time lags
            tmp = randperm(Npout); 
            outk0 = out_ind(tmp(1:Nsamout)); % out ky indices for 1st echo 
            kytable(Nnav+1:end,1,ii)=outk0;
            for jj = 1:Nsamout  % loop sampled out ky at 1st echo
                klim = outk0(jj)+kyr; % ky-range for this sample
                klim(klim<1)=1;  % ky must be within [1, Np]
                klim(klim>Np)=Np;
                kyecho = klim(1):klim(2); % ky list of echos
                nkyecho = length(kyecho);
                tmp = randi(nkyecho, 1, Necho-1); % select ky for other echos
                outke = kyecho(tmp); % out k-y indices for other echos
                for kk=1:Necho-1
                    kytable(Nnav+jj,kk+1,ii) = outke(kk);
                end
            end
        end
        navord = round(linspace(1,Nsam,Nnav)); % reorgnize navigator acquisition order       
        outord = (Nnav+1:Nsam)';
        neword = [1;outord];
        for ii = 2:Nnav      
            if navord(ii) < Nsam
                neword = [neword(1:navord(ii)-1);ii;neword(navord(ii):end)]; % move the navigator to new order
            else
                neword = [neword;ii];
            end
        end
        kytable = kytable(neword,:,:);
end
varargout{1} = nav_ind;