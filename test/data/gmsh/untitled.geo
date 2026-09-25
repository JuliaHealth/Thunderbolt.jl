//+
SetFactory("OpenCASCADE");
Disk(1) = {1.4, 0.8, 0, 0.5, 0.25};
//+
Disk(2) = {1.4, 0.8, 0, 2, 2};
//+
BooleanDifference{ Surface{2}; Delete; }{ Surface{1}; }

