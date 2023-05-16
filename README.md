# Cache-simulator

Design Decisions:
1. We must build a data structure to represent cache and related information 
of cache. 

2. Since our program only care about instruction of lw and sw which will load data from 
memory or store data into the memory, we only need to modify cache when we have 
lw and sw instruction.

3. When we reach lw instruction, we need to calculate corresponding tag and row.
According to the tag and row, we check whether cache contains the tag. If yes, hit.
If no, miss. If there are two cache and first cache miss, we can go to second 
cache to check whether the address has been stored in second cache. We only 
need to check two caches since our program only need to support up to 2 caches. 

4. When we reach sw instruction, we need to calculate corresponding tag and row.
According to the tag and row, we will store value in right place of cache. If there is more
than one cache, we also need to store the value into other cache with right row and tag.

5. Meanwhile, since we have associative cache, we can use LRU function to
keep track of the order in which blocks have been used. If the cache row is 
full, we need to evict the least recently used block to make room for new value.

The idea of LRU function is that I will store the just used block at the end of array. 
When LRU function has been called, every entry will move one index forward and new block 
will be added to the end of array, which means the first entry will be removed.

Strengths:
I try to avoid needless redundancy and the code are easy to understand with necessary comments. 
