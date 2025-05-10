while :; do
    echo transfering logs

    scp root@34.122.184.81:/mnt/logGames/log1    root@34.154.196.100:/root/logGames/log1
    scp root@34.122.184.81:/mnt/logGames/log2    root@34.154.196.100:/root/logGames/log2
    scp root@35.219.238.110:/root/logGames/log1  root@34.154.196.100:/root/logGames/log3
    scp root@35.219.238.110:/root/logGames/log2  root@34.154.196.100:/root/logGames/log4
    ssh root@34.154.196.100 "touch /root/done/doneLogs"

    echo waiting til modelWeights updates
    while ! scp root@34.154.196.100:/root/done/doneModelWeights /tmp/trash 2> /dev/null; do   
        sleep 0.5
    done
    ssh root@34.154.196.100 "rm /root/done/doneModelWeights"

    echo transfering modelWeights

    scp root@34.154.196.100:/root/tfgInformatica/backgammon/modelWeights  root@34.122.184.81:/mnt/tfgInformatica/backgammon/modelWeights1
    scp root@34.154.196.100:/root/tfgInformatica/backgammon/modelWeights  root@34.122.184.81:/mnt/tfgInformatica/backgammon/modelWeights2
    ssh root@34.122.184.81 "touch /mnt/done/doneWeights1"
    ssh root@34.122.184.81 "touch /mnt/done/doneWeights2"

    scp root@34.154.196.100:/root/tfgInformatica/backgammon/modelWeights  root@35.219.238.110:/root/tfgInformatica/backgammon/modelWeights1                 
    scp root@34.154.196.100:/root/tfgInformatica/backgammon/modelWeights  root@35.219.238.110:/root/tfgInformatica/backgammon/modelWeights2   
    ssh root@35.219.238.110 "touch /root/done/doneWeights1"       
    ssh root@35.219.238.110 "touch /root/done/doneWeights2"     

    echo waiting til partida ends
    while ! scp root@34.154.196.100:/root/done/donePartidaLarga /tmp/trash 2> /dev/null; do   
        sleep 2
    done

    echo transfering partida

    scp root@34.154.196.100:/root/tfgInformatica/backgammon/partida ./partidasLargas/$(date '+%H:%M,%d-%m').mat

done