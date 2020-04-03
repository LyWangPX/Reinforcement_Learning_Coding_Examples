## Review of A2C and A3C
We all know the A3C is using a episode count to control the async
process. In this procedure, child process will return the grad
after given episodes or finished early. They will be sync in the next
run. 

A2C however, waits for each child process to finish its segments.

## My Variation
In this project, I change the sync or async method from A2C and A3C
into episode wise. That is, the child process will only return 
the grad AFTER it completes the episode.

In variation of A3C, child process will return the grad as soon
as it finished the current episode, return the latest grad, do the backward
then sync with the latest model parameters to go next. It will be put
into an infinite loop. This method will never call join() method to process,
instead, it will monitor a queue that is filled by each child process's game
record. If the last 100 of them are maximum rewards, it will send terminate
message to all child processes.

In variation of A2C, child process will return the grad as soon 
as it finished the current episode but all processes will wait every one
finished and sync with the updated model together. In this case we need
a loop and call joint in each loop. If the queue fulfills the convergence requirement,
loop will be ended.

Additionally, I have set all mode do not learn at all if it reaches the maximum 
reward to facilitate converging. [need TESTED]

## More Variations:
There isn't too much difference in each child processes. It could 
be a exploratory directions. And, how to measure the differences of each 
method need a large amount of time.

## Additional Warnings:
when initialize multi-processing, fork does not work. If you are in Linux or Mac, change to spawn.
Reasons unclear to me at this point.
