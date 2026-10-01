#!/bin/bash
for j in 25422302 25422303 25422304 25422305 25422306 25422307; do
  echo "-- $j"
  qstat -f "$j" 2>/dev/null | grep -iE 'Resource_List.select|Resource_List.place|select =|place =' | head -3
done