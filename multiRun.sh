if [ "$#" -ne 2 ]; then
	echo "Usage: $0 <python_script> <repeates>"
	exit 1
fi

python_script="$1"
repeats="$2"

if [ ! -f "$python_script" ];then
	echo "Error: Python script '$python_script' not found."
	exit 1
fi

for i in $(seq 1 "$repeats"); do
	echo "Running '$python_script', iteration $i of $repeats ... "
	/home/u23/amesmaieeli/miniconda3/envs/pytorchProj_main/bin/python3.12 "$python_script"
	if [$? -ne 0]; then
		echo "Error: Python script '$python_script' exited wih a non-zero state"
		exit 1
	fi
done

echo "All repeats are done"
exit 0
