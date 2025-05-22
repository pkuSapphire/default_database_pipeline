
# Check for Famous Defaults

1. **Enron**: I find that it do have default ratings in 2001, but it does not have financial data in 2001, so we automatically drop the default. I have checked that it did not have [10-K](https://enroncorp.com/src/doc/investors/2001-04-02-10-k.pdf) for 2001, only [10-Q](https://www.sec.gov/Archives/edgar/data/1024401/000095012901504218/h92492e10-q.txt) and that's why we could not automatically include it.
   
2. **Kodak**: It is included in base_dataset, it defaulted at 2012-01-19. It had new ratings later on, yet we focus on the first default. I am sure it works fine in this version.

3. **Polaroid**: It has the same problem as Enron, it does not have financial data for 2001, when it defaulted.

4. **PG&E**: It is fixed, like Kodak.

5. **Worldcom**: We do have financial data for it, but we do not have it in ratings(entity rating).

