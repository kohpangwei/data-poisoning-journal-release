function prettyPrint(s)
  % pretty-prints a struct
  allFields = fields(s);
  for i=1:length(allFields)
    fprintf(1, '\t%8s:', allFields{i});
    disp(getfield(s, allFields{i}));
  end
end
