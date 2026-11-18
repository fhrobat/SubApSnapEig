%CONVERT  Convert the files saved by train_modernbert.py to the format read by
%   problems/problem_transformer.m. Run it in the folder with the files
%   step_<s>_ROPE.mat: the 1 x 5000 x 64 single-precision arrays Q_rope and
%   K_rope are replaced by 5000 x 64 double-precision matrices.

files = dir('step_*_ROPE.mat');
for i = 1:numel(files)
    data = load(files(i).name);
    for v = fieldnames(data)'
        data.(v{1}) = double(squeeze(data.(v{1})));
    end
    save(files(i).name, '-struct', 'data');
end
