#!/bin/bash

for measure in goodput cwnd_evolution
do
    for pep in none bbr hybla "fec" 
    do
		outpref=tcp"_"$pep
		grep $pep tcp_$measure.csv | awk -F, '{ print $1, $2, $3, $5, $6, $7, $8, $9 }' > $outpref"_"$measure.csv
    done
done

for prot in 'tcp' 'tls'
do
    for measure in time_to_first_byte connection_establishment 
    do
		for pep in none bbr hybla fec 
		do
			prefix=$prot"_"$pep
		    grep $pep $prot"_"$measure.csv | awk -F, '{ print $1, $2, $3, $5, $6, $7, $8 }' > $prefix"_"$measure.csv
		done
    done
done

fileset="tcp_none_connection_establishment tcp_bbr_connection_establishment tcp_hybla_connection_establishment tcp_fec_connection_establishment tls_none_connection_establishment tls_bbr_connection_establishment tls_hybla_connection_establishment tls_fec_connection_establishment quic_connection_establishment quic_fec_connection_establishment tcp_none_time_to_first_byte tcp_bbr_time_to_first_byte tcp_hybla_time_to_first_byte tcp_fec_time_to_first_byte tls_none_time_to_first_byte tls_bbr_time_to_first_byte tls_hybla_time_to_first_byte tls_fec_time_to_first_byte quic_time_to_first_byte quic_fec_time_to_first_byte"

# Plot files with a single data point per file

for file in $fileset
do
    sed -e /delay/d -e "s/,/ /g" -e "s/LEO/75/" -e "s/MEO/150/" -e "s/GEO/300/" -e "s/mbit//"  $file.csv | awk '{ if ($1 != owd) { print (""); owd = $1; } print; }' > $file.dat

    case $file in
	tls_none_*establishment*)
	    title="TLS connection establishment time"
	    ymax=3.5 # 3
	    yl=2.2
	    ;;
	tls_bbr_*establishment*)
	    title="TLS PEP (BBR) connection establishment time"
	    ymax=3.5 # 3
	    yl=2.2
	    ;;
	tls_hybla_*establishment*)
	    title="TLS PEP (Hybla) connection establishment time"
	    ymax=3.5 # 3
	    yl=2.2
	    ;;
	tls_fec_*establishment*)
	    title="TLS (FEC-tunnel) connection establishment time"
	    ymax=3.5 # 3
	    yl=2.2
	    ;;
	tcp_none*establishment*)
	    title="TCP connection establishment time"
	    ymax=2 # 1
	    yl=1.2
	    ;;
	tcp_bbr*establishment*)
	    title="TCP PEP (BBR) connection establishment time"
	    ymax=2 # 1
	    yl=1.2
	    ;;
	tcp_hybla*establishment*)
	    title="TCP PEP (Hybla) connection establishment time"
	    ymax=2 # 1
	    yl=1.2
	    ;;
	tcp_fec*establishment*)
	    title="TCP (FEC-tunnel) connection establishment time"
	    ymax=2 # 1
	    yl=1.2
	    ;;
	quic_connection_establishment*)
	    title="QUIC connection establishment time"
	    ymax=1 # 1
	    yl=0.1
	    ;;
	quic_fec_connection_establishment*)
	    title="QUIC (FEC-tunnel) connection establishment time"
	    ymax=1 # 1
	    yl=0.1
	    ;;
	tls_none*first_byte*)
	    title="TLS time to first byte"
	    ymax=6 # 4
	    yl=4
	    ;;
	tls_bbr*first_byte*)
	    title="TLS PEP (BBR) time to first byte"
	    ymax=6 # 4
	    yl=4
	    ;;
	tls_hybla*first_byte*)
	    title="TLS PEP (Hybla) time to first byte"
	    ymax=6 # 4
	    yl=4
	    ;;
	tls_fec*first_byte*)
	    title="TLS (FEC-tunnel) time to first byte"
	    ymax=6 # 4
	    yl=4
	    ;;
	tcp_none*first_byte*)
	    title="TCP time to first byte"
	    ymax=3 # 2
	    yl=2
	    ;;
	tcp_bbr*first_byte*)
	    title="TCP PEP (BBR) time to first byte"
	    ymax=3 # 2
	    yl=2
	    ;;
	tcp_hybla*first_byte*)
	    title="TCP PEP (Hybla) time to first byte"
	    ymax=3 # 2
	    yl=2
	    ;;
	tcp_fec*first_byte*)
	    title="TCP (FEC-tunnel) time to first byte"
	    ymax=3 # 2
	    yl=2
	    ;;
	quic_time_to_first_byte*)
	    title="QUIC time to first byte"
	    ymax=3 # 2
	    yl=2
	    ;;
	quic_fec_time_to_first_byte*)
	    title="QUIC (FEC-tunnel) time to first byte"
	    ymax=3 # 2
	    yl=2
	    ;;
	*)
	    title="Unknown * time"
	    ;;
    esac
	

    gnuplot <<EOF
set title "$title"
set key top left
set ylabel "time (s)"
set xlabel "Satellite type, link capacity (Mbit/s)"
set xrange [0:12]
set yrange [0:$ymax]
set label "LEO" at 2,$yl center
set label "MEO" at 6,$yl center
set label "GEO" at 10,$yl center
set term pdf size 8cm, 8cm
set output '$file.pdf'
set pointsize 1
set xtics 1
plot '$file.dat' every 4::0 using (log10(\$2)+1+4*(int(\$1/75) >> 1)-0.3):4:5:7 with errorbars pt 3 lc "black" title '0.01%', \
              '' every 4::1 using (log10(\$2)+1+4*(int(\$1/75) >> 1)-0.1):4:5:7:xtic(2) with errorbars pt 4 lc "blue" title '0.1%', \
              '' every 4::2 using (log10(\$2)+1+4*(int(\$1/75) >> 1)+0.1):4:5:7 with errorbars pt 8 lc "red" title '1.0%', \
              '' every 4::3 using (log10(\$2)+1+4*(int(\$1/75) >> 1)+0.3):4:5:7 with errorbars pt 9 lc "purple" title '5.0%'
EOF

done

# Plot files with a single data point per file
fileset="tcp_none_goodput tcp_bbr_goodput tcp_hybla_goodput tcp_fec_goodput tcp_none_cwnd_evolution tcp_bbr_cwnd_evolution tcp_hybla_cwnd_evolution tcp_fec_cwnd_evolution quic_goodput quic_fec_goodput quic_cwnd_evolution quic_fec_cwnd_evolution" 

for file in $fileset
do
    sed -e /delay/d -e "s/,/ /g" -e "s/ms//" -e "s/mbit//" $file.csv | awk '{ if ($1 != owd) { print (""); owd = $1; } if ($4 < 30) print; }' > $file.dat
    
    case $file in
    tcp_none*goodput*)
	    title="TCP goodput"
	    ylabel="Goodput (kbps)"
	    ;;
	tcp_none*cwnd*)
	    title="TCP CWND"
	    ylabel="Congestion window (KB)"
	    ;;
	quic_goodput*)
	    title="QUIC goodput"
	    ylabel="Goodput (kbps)"
	    ;;
	quic_fec_goodput*)
	    title="QUIC (FEC-tunnel) goodput"
	    ylabel="Goodput (kbps)"
	    ;;
	quic_cwnd*)
	    title="QUIC CWND"
	    ylabel="Congestion window (KB)"
	    ;;
	quic_fec_cwnd*)
	    title="QUIC CWND (FEC-tunnel)"
	    ylabel="Congestion window (KB)"
	    ;;
	tcp_bbr*goodput*)
	    title="TCP PEP (BBR) goodput"
	    ylabel="Goodput (kbps)"
	    ;;
	tcp_bbr*cwnd*)
	    title="TCP PEP (BBR) CWND"
	    ylabel="Congestion window (KB)"
	    ;;
	tcp_hybla*goodput*)
	    title="TCP PEP (Hyble) goodput"
	    ylabel="Goodput (kbps)"
	    ;;
	tcp_hybla*cwnd*)
	    title="TCP PEP (Hybla) CWND"
	    ylabel="Congestion window (KB)"
	    ;;
	tcp_fec*goodput*)
	    title="TCP (FEC-tunnel) goodput"
	    ylabel="Goodput (kbps)"
	    ;;
	tcp_fec*cwnd*)
	    title="TCP (FEC-tunnel) CWND"
	    ylabel="Congestion window (KB)"
	    ;;
	*)
	    title="Unknown content"
	    ylabel="Unknown"
	    ;;
    esac


    for sat in LEO MEO GEO
    do
	case $sat in
	    LEO)
		block=0
		;;
	    MEO)
		block=1
		;;
	    GEO)
		block=2
		;;
	esac
	gnuplot <<EOF
set title "$title - $sat"
set key right outside center vertical
set ylabel "$ylabel"
set xlabel "Time (s)"
set xrange [0:30]
set term pdf size 12cm, 6cm
set output '$file-$sat.pdf'
set pointsize 1
plot '$file.dat' every 1::0:$block:29:$block using   (\$4+1):(\$5/1000) with linespoints pt 2 lc "black" title '1 Mbps, p=0.01%', \
              '' every 1::30:$block:59:$block using (\$4+1):(\$5/1000) with linespoints pt 4 lc "black" title '1 Mbps, p=0.1%', \
              '' every 1::60:$block:89:$block using (\$4+1):(\$5/1000) with linespoints pt 8 lc "black" title '1 Mbps, p=1.0%', \
              '' every 1::90:$block:119:$block using (\$4+1):(\$5/1000) with linespoints pt 10 lc "black" title '1 Mbps, p=5.0%', \
              '' every 1::120:$block:149:$block using (\$4+1):(\$5/1000) with linespoints pt 2 lc "blue" title '10 Mbps, p=0.01%', \
              '' every 1::150:$block:179:$block using (\$4+1):(\$5/1000) with linespoints pt 4 lc "blue" title '10 Mbps, p=0.1%', \
              '' every 1::180:$block:209:$block using (\$4+1):(\$5/1000) with linespoints pt 8 lc "blue" title '10 Mbps, p=1.0%', \
              '' every 1::210:$block:239:$block using (\$4+1):(\$5/1000) with linespoints pt 10 lc "blue" title '10 Mbps, p=5.0%', \
              '' every 1::240:$block:269:$block using (\$4+1):(\$5/1000) with linespoints pt 2 lc "red" title '100 Mbps, p=0.01%', \
              '' every 1::270:$block:299:$block using (\$4+1):(\$5/1000) with linespoints pt 4 lc "red" title '100 Mbps, p=0.1%', \
              '' every 1::300:$block:329:$block using (\$4+1):(\$5/1000) with linespoints pt 8 lc "red" title '100 Mbps, p=1.0%', \
              '' every 1::330:$block:359:$block using (\$4+1):(\$5/1000) with linespoints pt 10 lc "red" title '100 Mbps, p=5.0%'
EOF

    done

done

########### Compare QUIC and TCP in a single graph
#
# we rely on having done the SED process on these files in the previous step

fileset="goodput cwnd_evolution" 

for file in $fileset
do
    case $file in
	goodput)
	    title="Goodput evolution"
	    ylabel="Goodput (kbps)"
	    ;;
	cwnd*)
	    title="Congestion window evolution"
	    ylabel="Congestion window (KB)"
	    ;;
	*)
	    title="Unknown content"
	    ylabel="Unknown"
	    ;;
    esac

    for sat in LEO MEO GEO
    do
	case $sat in
	    LEO)
		block=0
		;;
	    MEO)
		block=1
		;;
	    GEO)
		block=2
		;;
	esac

	for rate in 1 10 100
	do
	    case $rate in
		1)
		    offset=0
		    ;;
		10)
		    offset=120
		    ;;
		100)
		    offset=240
		    ;;
	    esac
	    gnuplot <<EOF
set title "$title - $sat - $rate Mbit/s"
set key outside right center vertical
set ylabel "$ylabel"
set xlabel "Time (s)"
set xrange [0:30]
set term pdf size 16cm, 9cm
set output '$file-$sat-$rate.pdf'
set pointsize 1
set logscale y
plot 'tcp_none_$file.dat' every 1::$offset+0:$block:$offset+29:$block using  (\$4+1):(\$5/1000) with linespoints pt 2 lc "black" title 'TCP p=0.01%', \
              '' every 1::$offset+30:$block:$offset+59:$block using (\$4+1):(\$5/1000) with linespoints pt 4 lc "black" title 'TCP p=0.1%', \
              '' every 1::$offset+60:$block:$offset+89:$block using (\$4+1):(\$5/1000) with linespoints pt 8 lc "black" title 'TCP p=1.0%', \
              '' every 1::$offset+90:$block:$offset+119:$block using (\$4+1):(\$5/1000) with linespoints pt 10 lc "black" title 'TCP p=5.0%', \
     'tcp_bbr_$file.dat' every 1::$offset+0:$block:$offset+29:$block using  (\$4+1):(\$5/1000) with linespoints pt 2 lc "red" title 'TCP PEP (BBR) p=0.01%', \
              '' every 1::$offset+30:$block:$offset+59:$block using (\$4+1):(\$5/1000) with linespoints pt 4 lc "red" title 'TCP PEP (BBR) p=0.1%', \
              '' every 1::$offset+60:$block:$offset+89:$block using (\$4+1):(\$5/1000) with linespoints pt 8 lc "red" title 'TCP PEP (BBR) p=1.0%', \
              '' every 1::$offset+90:$block:$offset+119:$block using (\$4+1):(\$5/1000) with linespoints pt 10 lc "red" title 'TCP PEP (BBR) p=5.0%', \
     'tcp_fec_$file.dat' every 1::$offset+0:$block:$offset+29:$block using  (\$4+1):(\$5/1000) with linespoints pt 2 lc "dark-violet" title 'TCP (FEC-tunnel) p=0.01%', \
              '' every 1::$offset+30:$block:$offset+59:$block using (\$4+1):(\$5/1000) with linespoints pt 4 lc "dark-violet" title 'TCP (FEC-tunnel) p=0.1%', \
              '' every 1::$offset+60:$block:$offset+89:$block using (\$4+1):(\$5/1000) with linespoints pt 8 lc "dark-violet" title 'TCP (FEC-tunnel) p=1.0%', \
              '' every 1::$offset+90:$block:$offset+119:$block using (\$4+1):(\$5/1000) with linespoints pt 10 lc "dark-violet" title 'TCP (FEC-tunnel) p=5.0%', \
     'quic_$file.dat' every 1::$offset+0:$block:$offset+29:$block using  (\$4+1):(\$5/1000) with linespoints pt 2 lc "blue" title 'QUIC p=0.01%', \
              '' every 1::$offset+30:$block:$offset+59:$block using (\$4+1):(\$5/1000) with linespoints pt 4 lc "blue" title 'QUIC p=0.1%', \
              '' every 1::$offset+60:$block:$offset+89:$block using (\$4+1):(\$5/1000) with linespoints pt 8 lc "blue" title 'QUIC p=1.0%', \
              '' every 1::$offset+90:$block:$offset+119:$block using (\$4+1):(\$5/1000) with linespoints pt 10 lc "blue" title 'QUIC p=5.0%', \
     'quic_fec_$file.dat' every 1::$offset+0:$block:$offset+29:$block using  (\$4+1):(\$5/1000) with linespoints pt 2 lc "dark-green" title 'QUIC (FEC-tunnel) p=0.01%', \
              '' every 1::$offset+30:$block:$offset+59:$block using (\$4+1):(\$5/1000) with linespoints pt 4 lc "dark-green" title 'QUIC (FEC-tunnel) p=0.1%', \
              '' every 1::$offset+60:$block:$offset+89:$block using (\$4+1):(\$5/1000) with linespoints pt 8 lc "dark-green" title 'QUIC (FEC-tunnel) p=1.0%', \
              '' every 1::$offset+90:$block:$offset+119:$block using (\$4+1):(\$5/1000) with linespoints pt 10 lc "dark-green" title 'QUIC (FEC-tunnel) p=5.0%'

EOF
	done   # $rate
    done       # $sat
done           # $file

#       plot 'file' every {<point_incr>}
#                           {:{<block_incr>}
#                             {:{<start_point>}
#                               {:{<start_block>}
#                                 {:{<end_point>}
#                                   {:<end_block>}}}}}
#       every :::3::3    # selects just the fourth block ('0' is first)
#       every :::::9     # selects the first 10 blocks
#       every 2:2        # selects every other point in every other block
#       every ::5::15    # selects points 5 through 15 in each block

