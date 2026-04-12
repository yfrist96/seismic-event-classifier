#!/usr/bin/env bash

Format='ANTELOPE'
dbname='/Users/ik_seislab/Accounts/ittaik/ReprocCat_2013-2019/TransferLearning_2013-2019/Reprocessing_022024/Combine_2013-2019/RelocEQtransformer_2013-2019_SNR25' # CHANGE name of DB
Network='IS'
WF_dir='/Users/ik_seislab/Accounts/ittaik/ReprocCat_2013-2019/WF/wfs_ydns'
Output_400='/Users/ik_seislab/Accounts/ittaik/ReprocCat_2013-2019/TransferLearning_2013-2019/Trans2013-2019_SNR25/GrowClustResults/400_2013-2019_050524_Reloc/' # CHANGE
#Output_410='/Users/ik_seislab/Accounts/ittaik/ReprocCat_2013-2019/TransferLearning_2013-2019/Trans2013-2019_SNR25/GrowClustResults/410_2013-2019_050524_Reloc' # CHANGE

src_400='/Users/ik_seislab/Accounts/ittaik/ReprocCat_2013-2019/TransferLearning_2013-2019/Trans2013-2019_SNR25/GrowClustResults/400_extract_event_wfs_antelope.py' # UPDATE
src_410='/Users/ik_seislab/Accounts/ittaik/ReprocCat_2013-2019/TransferLearning_2013-2019/Trans2013-2019_SNR25/GrowClustResults/410_measure_differentials_antelope.py' # UPDATE

file="/Users/ik_seislab/Accounts/ittaik/ReprocCat_2013-2019/TransferLearning_2013-2019/Trans2013-2019_SNR25/GrowClustResults/sta_2013-2019.txt" # CHECK Station List
# file="StationList_4REF_CAT.txt"
Stations=`cat $file`
for Station in $Stations; do
        echo "$Station"
        python $src_400 -f $Format $dbname $Network $Station $WF_dir $Output_400
        #python $src_410 -f $Format $dbname $Network $Station $Output_400 $Output_410 
done

echo All done


# python 420_reformat_cc_output.py -f ANTELOPE...
#  /Users/ittaik/Dropbox/MyKurzon/db/ReprocessingCatalogue/2013-2019_Analysis/Boston2021_tests/EQtransformer_testing/Mac2Mac-08122022/DSF-Picks-DB-20180601-20180801-REF4CAT-ndf4-SNR25/RelocDSF_Picks_REF4CAT_ndf4_SNR25/RelocDSF_Picks_REF4CAT_ndf4_SNR25...
#  /Volumes/easystore/GrowClustResults/410_4ML_testing_20180601-20180801/...
#  /Volumes/easystore/GrowClustResults/420_4ML_testing_20180601-20180801
