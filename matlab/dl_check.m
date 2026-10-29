function ok = dl_check(ok, cond, name)
%DL_CHECK  Helper for run_tests.
if cond, fprintf('PASS  %s\n', name); else, fprintf('FAIL  %s\n', name); ok = false; end
end
