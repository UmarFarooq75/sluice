# Canonical preflight primitives. SOURCE this; do not copy it.
#
# Why it exists: every experiment used to hand-roll its own avail check, and they
# drifted. E37b/c summed four vm_stat buckets with the page size queried from
# vm_stat; results/e41b/gate.py summed three with the page size hardcoded to 4096.
# Two yardsticks means a run can pass one gate and abort on the other, and the
# 4096 assumption is itself the bug that once read 2.58 GB on a machine with
# 10.7 GB free (Apple Silicon pages are 16384 bytes).
#
# One definition, one place. New rungs source this file.

# Free-ish RAM in GB. Page size QUERIED from vm_stat, never assumed.
sl_avail_gb() {
  vm_stat | awk '
    /page size of/ { for (i=1;i<=NF;i++) if ($i=="of") ps=$(i+1) }
    /Pages free/{f=$3} /Pages inactive/{iv=$3} /Pages speculative/{sp=$3} /Pages purgeable/{pu=$3}
    END { gsub(/\./,"",f); gsub(/\./,"",iv); gsub(/\./,"",sp); gsub(/\./,"",pu);
          printf "%.2f", (f+iv+sp+pu)*ps/1e9 }'
}

# GB of swap currently in use. "8 GB free" on a box already paging is not 8 GB free.
sl_swap_gb() {
  sysctl -n vm.swapusage 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="used"){gsub(/M/,"",$(i+2)); printf "%.2f", $(i+2)/1024; exit}}'
}

# protocol #1: exactly one model process on this machine, ever.
sl_engine_running() { pgrep -f "csrc/stream_run" > /dev/null 2>&1; }

# numeric >= without bc
sl_ge() { awk "BEGIN{exit !($1 >= $2)}"; }
