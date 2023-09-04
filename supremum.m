function sup = supremum(nums)
    if isempty(nums)
        sup = NaN;  % Handle the case of an empty set
        return;
    end
    
    % Find the maximum element in the array
    sup = max(nums);
end


