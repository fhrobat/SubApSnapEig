function write_dat(file, data, names)
%WRITE_DAT  Write a numeric matrix as a tab-separated file with a header row.
writetable(array2table(data, 'VariableNames', cellstr(names)), file, ...
           'Delimiter', '\t', 'FileType', 'text');
end
