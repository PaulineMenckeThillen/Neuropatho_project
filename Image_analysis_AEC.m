
%% --- INTELLIGENT SENSITIVITY FUNCTION ---
function [statsTable, reportImg] = Image_analysis_AEC(img)
    % 1. Noise Filter
    img_filt = cat(3, medfilt2(img(:,:,1), [3 3]), ...
                      medfilt2(img(:,:,2), [3 3]), ...
                      medfilt2(img(:,:,3), [3 3]));
    
    I_double = double(img_filt) / 255; I_double = max(I_double, 0.001); 
    OD = -log10(I_double);
    
    % 2. Deconvolution
    M = [0.650, 0.704, 0.286; 0.268, 0.570, 0.776; 0.286, 0.105, 0.952];  
    C = reshape(OD, [], 3) * pinv(M);
    H_chan = max(0, reshape(C(:,1), size(img,1), size(img,2)));
    AEC_chan = max(0, reshape(C(:,2), size(img,1), size(img,2)));
    
    % 3. Nuclei Detection
    H_flat = imtophat(H_chan, strel('disk', 15));
    nuc_level = max(0.05, graythresh(H_flat) * 0.85);
    nucMask = bwareaopen(H_flat > nuc_level, 50);

    % 4. DYNAMIC AEC THRESHOLDING
    AEC_filtered = imtophat(AEC_chan, strel('disk', 20));
    
    % Calculate local noise to set threshold
    noise_floor = prctile(AEC_filtered(:), 50); 
    noise_std = std(AEC_filtered(:));
    
    % Adaptive thresholding above the noise floor
    dynamic_thresh = max(0.075, noise_floor + (3.2 * noise_std));
    aecMask = AEC_filtered > dynamic_thresh;
    
    % Artifact Filtering
    if any(aecMask(:))
        aecMask = bwpropfilt(aecMask, 'Area', [15, 1500]); 
        aecMask = bwpropfilt(aecMask, 'Solidity', [0, 0.92]); 
    end

    % 5. Stats
    statsTable = table();
    stats = regionprops(nucMask, 'Area', 'PixelIdxList', 'Circularity');
    tempT = struct2table(stats);
    if ~isempty(tempT)
        tempT = tempT(tempT.Circularity > 0.35, :);
        numObjs = height(tempT);
        nAEC = zeros(numObjs, 1); cAEC = zeros(numObjs, 1);
        se = strel('disk', 6);
        for i = 1:numObjs
            idx = tempT.PixelIdxList{i};
            nAEC(i) = mean(AEC_chan(idx));
            tM = false(size(AEC_chan)); tM(idx) = true;
            cytoM = (imdilate(tM, se) & ~tM);
            if any(cytoM(:)), cAEC(i) = mean(AEC_chan(cytoM)); end
        end
        tempT.Nuclear_AEC = nAEC; tempT.Cytoplasmic_AEC = cAEC;
        statsTable = removevars(tempT, 'PixelIdxList');
    end
    
    % 6. Visual Output
    r = img(:,:,1); r(aecMask) = 255; 
    g = img(:,:,2); g(aecMask) = 0; 
    b = img(:,:,3); b(aecMask) = 0;
    reportImg = [img, repmat(uint8(nucMask*255), [1 1 3]), cat(3,r,g,b)];
end