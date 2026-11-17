function write_times(file, t)
%WRITE_TIMES  Write the timings of LU, Leverage and MOR (rows of t = [offline, online]).
%   The columns are those of the tables in the paper; the ratios are taken
%   with respect to MOR (called POD in the file, as in the original files).
Metodo = {'LU'; 'Leverage'; 'POD'};
tot = sum(t, 2);
T = table(Metodo, t(:,1), t(:,2), tot, t(:,1)/t(3,1), t(:,2)/t(3,2), tot/tot(3), ...
    'VariableNames', {'Metodo', 'Offline_sec', 'Online_sec', 'Totale_sec', ...
                      'Ratio_Off_vs_POD', 'Ratio_On_vs_POD', 'Ratio_vs_POD'});
writetable(T, file, 'Delimiter', '\t', 'FileType', 'text');
disp(T);
end
