clc; clear; close all;

%% --- 1. SETTINGS ---
inputRoot = 'D:\QuPathExportGFAP\tiles'; 
outputCSV = fullfile(inputRoot, 'GFAP_DAB_Analysis_Results.csv');

% Filter for target folders containing GFAP
dirFolders = dir(fullfile(inputRoot, '*GFAP*'));
dirFolders = dirFolders([dirFolders.isdir]); 

% RAM Optimization Setting: Write data to disk and clear cache every X tiles
saveFrequency = 50; 

%% --- 2. THE BATCH PROCESS ---
totalFolders = length(dirFolders);
firstWrite = ~exist(outputCSV, 'file'); % Track if we need to write fresh headers

for f = 1:totalFolders
    currentFolderName = dirFolders(f).name;
    currentFolderPath = fullfile(inputRoot, currentFolderName);
    
    caseSplit = strsplit(currentFolderName, '.czi');
    caseName = caseSplit{1};
    
    % Gather ALL tiles inside the folder
    tileFiles = dir(fullfile(currentFolderPath, '*.tif'));
    numTiles = length(tileFiles);
    
    % Skip folder if it contains no images
    if isempty(tileFiles)
        fprintf('\n[%d/%d] Folder: %s (No .tif files found - skipping)\n', f, totalFolders, currentFolderName);
        continue;
    end
    
    fprintf('\n[%d/%d] Folder: %s (%d tiles total)\n', f, totalFolders, currentFolderName, numTiles);
    
    % Generate local overlay folder path
    localOverlayFolder = fullfile(currentFolderPath, 'Analyzed_Overlays');
    if ~exist(localOverlayFolder, 'dir'), mkdir(localOverlayFolder); end
    
    % Initialize folder data table chunk
    ChunkTable = table();
    
    for k = 1:numTiles
        targetFile = tileFiles(k).name;
        fullFileName = fullfile(currentFolderPath, targetFile);
        
        try
            % Load and execute optimized DAB process
            img = imread(fullFileName);
            [currentTable, clinicalReport] = Image_analysis_DAB_QC_TEST(img);
            clear img; % Immediate RAM cleanup
            
            if ~isempty(currentTable)
                currentTable.CaseID = repmat({caseName}, height(currentTable), 1);
                currentTable.ScanRegion = repmat({currentFolderName}, height(currentTable), 1);
                currentTable.ImageName = repmat({targetFile}, height(currentTable), 1);
                ChunkTable = [ChunkTable; currentTable]; 
                clear currentTable;
            end
            
            % Save high-sensitivity visual report panel
            [~, name, ext] = fileparts(targetFile);
            imwrite(clinicalReport, fullfile(localOverlayFolder, [name '_DAB_Report' ext]));
            clear clinicalReport; % Immediate RAM cleanup
            
            % --- 3. MID-FOLDER RAM CLEARING ---
            % Saves data and clears cache dynamically during large folder sweeps
            if mod(k, saveFrequency) == 0 && ~isempty(ChunkTable)
                if firstWrite
                    writetable(ChunkTable, outputCSV, 'WriteMode', 'overwrite');
                    firstWrite = false;
                else
                    writetable(ChunkTable, outputCSV, 'WriteMode', 'append', 'WriteVariableNames', false);
                end
                ChunkTable = table(); % Empty the table to completely free up system RAM
                fprintf('...tiles %d/%d saved to disk (RAM cleared)\n', k, numTiles);
            end
            
        catch ME
            fprintf('  ! Error processing file %s: %s\n', targetFile, ME.message);
        end
    end
    
    % --- 4. FINAL FOLDER CLEANUP ---
    % Write out remaining tiles left in the ChunkTable for this folder
    if ~isempty(ChunkTable)
        if firstWrite
            writetable(ChunkTable, outputCSV, 'WriteMode', 'overwrite');
            firstWrite = false;
        else
            writetable(ChunkTable, outputCSV, 'WriteMode', 'append', 'WriteVariableNames', false);
        end
    end
    
    clear ChunkTable; % Drop final block from memory
    fprintf('  -> Folder %d pipeline complete.\n', f);
end

fprintf('\n--- BATCH PROCESS COMPLETE ---\nAll summary data matrices saved to: %s\n', outputCSV);