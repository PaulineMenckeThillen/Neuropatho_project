clc; clear; close all;

%% --- 1. SETTINGS ---
inputRoot = 'D:\QuPathExport\tiles'; 
outputCSV = fullfile(inputRoot, 'AEC_Analysis_Results.csv');

% Filter for folders
dirFolders = dir(fullfile(inputRoot, '*AEC*'));
dirFolders = dirFolders([dirFolders.isdir]); 

% RAM Optimization Setting: Write to disk every X tiles
saveFrequency = 50; 

%% --- 2. THE BATCH PROCESS ---
totalFolders = length(dirFolders);
firstWrite = ~exist(outputCSV, 'file'); % Check if we need to write headers

for f = 1:totalFolders
    currentFolderName = dirFolders(f).name;
    currentFolderPath = fullfile(inputRoot, currentFolderName);
    
    caseSplit = strsplit(currentFolderName, '.czi');
    caseName = caseSplit{1};
    
    tileFiles = dir(fullfile(currentFolderPath, '*.tif'));
    numTiles = length(tileFiles);
    
    fprintf('\n[%d/%d] Folder: %s (%d tiles)\n', f, totalFolders, currentFolderName, numTiles);
    
    localOverlayFolder = fullfile(currentFolderPath, 'Analyzed_Overlays');
    if ~exist(localOverlayFolder, 'dir'), mkdir(localOverlayFolder); end
    
    % Initialize a temporary table for small chunks
    ChunkTable = table();
    
    for k = 1:numTiles
        targetFile = tileFiles(k).name;
        fullFileName = fullfile(currentFolderPath, targetFile);
        
        try
            % Load and process
            img = imread(fullFileName);
            [currentTable, clinicalReport] = Image_analysis_AEC_QC_TEST(img);
            clear img; % RAM OPTIMIZATION: Clear raw image immediately after analysis
            
            if ~isempty(currentTable)
                currentTable.CaseID = repmat({caseName}, height(currentTable), 1);
                currentTable.ScanRegion = repmat({currentFolderName}, height(currentTable), 1);
                currentTable.ImageName = repmat({targetFile}, height(currentTable), 1);
                ChunkTable = [ChunkTable; currentTable]; 
                clear currentTable;
            end
            
            % Save visual report
            [~, name, ext] = fileparts(targetFile);
            imwrite(clinicalReport, fullfile(localOverlayFolder, [name '_Report' ext]));
            clear clinicalReport; % RAM OPTIMIZATION: Clear report image immediately after saving
            
            % --- 3. MID-FOLDER RAM CLEARING ---
            % Instead of waiting for the whole folder, save every X tiles
            if mod(k, saveFrequency) == 0 && ~isempty(ChunkTable)
                if firstWrite
                    writetable(ChunkTable, outputCSV, 'WriteMode', 'overwrite');
                    firstWrite = false;
                else
                    writetable(ChunkTable, outputCSV, 'WriteMode', 'append', 'WriteVariableNames', false);
                end
                ChunkTable = table(); % Empty the table to free RAM
                fprintf('...tiles %d/%d saved to disk (RAM cleared)\n', k, numTiles);
            end
            
        catch ME
            fprintf('  ! Error on %s: %s\n', targetFile, ME.message);
        end
    end
    
    % --- 4. FINAL FOLDER CLEANUP ---
    % Write any remaining tiles left in ChunkTable after the loop finishes
    if ~isempty(ChunkTable)
        if firstWrite
            writetable(ChunkTable, outputCSV, 'WriteMode', 'overwrite');
            firstWrite = false;
        else
            writetable(ChunkTable, outputCSV, 'WriteMode', 'append', 'WriteVariableNames', false);
        end
    end
    
    clear ChunkTable; % Final clear for the folder
    fprintf('  -> Folder %d complete.\n', f);
end

fprintf('\n--- BATCH COMPLETE ---\nResults: %s\n', outputCSV);