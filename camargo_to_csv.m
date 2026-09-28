% Convert the Camargo et al. (2021) MATLAB tables to CSV for the Python pipeline.
% Usage (from a shell):
%   matlab -batch "camargo_to_csv('<data>/Camargo2021/raw', '<data>/Camargo2021/csv')"
% Converts the sensors the analysis reads: markers, fp, conditions and id. The subject table
% SubjectInfo.mat (age, sex, height, mass), which download_camargo.py saves next to the raw folder,
% becomes SubjectInfo.csv next to the csv folder.

function camargo_to_csv(srcRoot, dstRoot)
    sensors = {'markers', 'fp', 'conditions', 'id'};
    modes = {'treadmill', 'ramp', 'levelground'};
    subs = dir(fullfile(srcRoot, 'AB*'));
    fprintf('subjects: %d\n', numel(subs));
    for i = 1:numel(subs)
        subj = subs(i).name;
        dates = dir(fullfile(srcRoot, subj));
        dates = dates([dates.isdir] & ~startsWith({dates.name}, '.'));
        for dd = 1:numel(dates)
            for mm = 1:numel(modes)
                for ss = 1:numel(sensors)
                    inDir = fullfile(srcRoot, subj, dates(dd).name, modes{mm}, sensors{ss});
                    if ~isfolder(inDir), continue; end
                    files = dir(fullfile(inDir, '*.mat'));
                    outDir = fullfile(dstRoot, subj, modes{mm}, sensors{ss});
                    if ~isfolder(outDir), mkdir(outDir); end
                    for ff = 1:numel(files)
                        outFile = fullfile(outDir, strrep(files(ff).name, '.mat', '.csv'));
                        if isfile(outFile), continue; end
                        S = load(fullfile(inDir, files(ff).name));
                        fn = fieldnames(S);
                        T = [];
                        for k = 1:numel(fn)
                            if istable(S.(fn{k})), T = S.(fn{k}); break; end
                        end
                        if isempty(T), continue; end
                        writetable(T, outFile);
                    end
                end
            end
        end
        fprintf('done %s\n', subj);
    end

    info = fullfile(fileparts(strip(srcRoot, 'right', filesep)), 'SubjectInfo.mat');
    if isfile(info)
        S = load(info);
        fn = fieldnames(S);
        writetable(S.(fn{1}), fullfile(fileparts(strip(dstRoot, 'right', filesep)), 'SubjectInfo.csv'));
        fprintf('wrote SubjectInfo.csv\n');
    else
        fprintf('SubjectInfo.mat not found next to %s\n', srcRoot);
    end
end
