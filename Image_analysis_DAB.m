%% --- INTELLIGENT SENSITIVITY FUNCTION ---
function [statsTable, reportImg] = Image_analysis_DAB(img)
    % 1. Noise Filter
    img_filt = cat(3, medfilt2(img(:,:,1), [3 3]), ...
                      medfilt2(img(:,:,2), [3 3]), ...
                      medfilt2(img(:,:,3), [3 3]));
    
    I_double = double(img_filt) / 255; I_double = max(I_double, 0.001); 
    OD = -log10(I_double);
    
    % 2. Deconvolution (Row 1: Hematoxylin, Row 2: DAB)
    M = [0.650, 0.704, 0.286; 0.268, 0.570, 0.776; 0.286, 0.105, 0.952];  
    C = reshape(OD, [], 3) * pinv(M);
    H_chan = max(0, reshape(C(:,1), size(img,1), size(img,2)));
    DAB_chan = max(0, reshape(C(:,2), size(img,1), size(img,2)));
    
    % 3. Nuclei Detection
    H_flat = imtophat(H_chan, strel('disk', 15));
    nuc_level = max(0.05, graythresh(H_flat) * 0.85);
    nucMask = bwareaopen(H_flat > nuc_level, 50);

    % 4. DYNAMIC DAB THRESHOLDING (High Sensitivity for GFAP Meshwork)
    % Increased disk size to 50 to avoid subtracting dense, localized GFAP clusters
    DAB_filtered = imtophat(DAB_chan, strel('disk', 50));
    
    % Calculate local noise floor
    noise_floor = prctile(DAB_filtered(:), 40); 
    noise_std = std(DAB_filtered(:));
    
    % Tuned multiplier (1.2) to pick up fine, branching processes
    dynamic_thresh = max(0.04, noise_floor + (1.2 * noise_std));
    dabMask = DAB_filtered > dynamic_thresh;
    
    % Artifact & Morphology Filtering
    if any(dabMask(:))
        % Maximum area is set to Inf because GFAP networks can form massive interconnected structures
        dabMask = bwpropfilt(dabMask, 'Area', [15, Inf]); 
        
        % Slightly relaxed solidity constraint to allow branched, non-convex astrocyte shapes
        dabMask = bwpropfilt(dabMask, 'Solidity', [0, 0.95]); 
    end

    % 5. Stats Extraction
    statsTable = table();
    stats = regionprops(nucMask, 'Area', 'PixelIdxList', 'Circularity');
    tempT = struct2table(stats);
    if ~isempty(tempT)
        tempT = tempT(tempT.Circularity > 0.35, :);
        numObjs = height(tempT);
        nDAB = zeros(numObjs, 1); cDAB = zeros(numObjs, 1);
        se = strel('disk', 6);
        for i = 1:numObjs
            idx = tempT.PixelIdxList{i};
            nDAB(i) = mean(DAB_chan(idx));
            tM = false(size(DAB_chan)); tM(idx) = true;
            cytoM = (imdilate(tM, se) & ~tM);
            if any(cytoM(:)), cDAB(i) = mean(DAB_chan(cytoM)); end
        end
        tempT.Nuclear_DAB = nDAB; tempT.Cytoplasmic_DAB = cDAB;
        statsTable = removevars(tempT, 'PixelIdxList');
    end
    
    % 6. Visual Output (DAB Overlay shown in Red)
    r = img(:,:,1); r(dabMask) = 255; 
    g = img(:,:,2); g(dabMask) = 0; 
    b = img(:,:,3); b(dabMask) = 0;
    reportImg = [img, repmat(uint8(nucMask*255), [1 1 3]), cat(3,r,g,b)];
end