/**
	@file
	multi_track
	
	multi_track is a MAX/MSP extension, the object that can load and run pytorvh neural network through running the Python server. 
	The the python server will be acquisitively fast and will use the GPU if available. 
	For the object to function, you need Python and Tensorflow installed on your machine. Ecxact list of nessesary librieries will be provided
	with the object in README file. multi_track takes the network's name, input matrix sizes and input data as arguments. 
	The object can load a neural network with message "read" saved in HDF5 format. The message with the message "data" followed by a list 
	of numbers and ending with mnessage "end" will trigger the prediction process. In response, the object will output a list of numbers 
	with indexies of the corresponding size as a prediction from the network. When deleted from the patch, the object must deactivate 
	the python server and free all the memory taken.

	Tornike Karchkhadze, tkarchkhadze@ucsd.edu
*/




#include "ext.h"
#include "ext_obex.h"
#include "jit.common.h"

#include <time.h>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <stdio.h>		/* for dumpping memroy to file */
#include <thread>
#include <chrono> // For optional delays

#ifdef _WIN32
#include "shellapi.h"
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")  // Link Winsock library
#include <tlhelp32.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#else
// macOS / POSIX
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <sys/time.h>
#include <sys/stat.h>
#endif

#include "osc/OscOutboundPacketStream.h"
#include <ip/IpEndpointName.h>
#include "osc/OscReceivedElements.h"
#include "osc/OscPacketListener.h"

#include "z_dsp.h"			// required for MSP objects



//using namespace std::placeholders;


//#define ADDRESS "reach2.ircam.fr" //"129.102.15.13" /* local host ip */
#define OUTPUT_BUFFER_SIZE 65536 /* maximum buffer size for osc send */
#define MAX_OSC_PACKET_SIZE 65535  // Maximum UDP packet size allowed
//#define NUM_SAMPLES 163840

// TCP replacement for TcpTransmitSocket — same interface, sends OSC over TCP with 4-byte length prefix
class TcpTransmitSocket {
	char ip[64];
	int port;
public:
	TcpTransmitSocket(IpEndpointName endpoint) {
		endpoint.AddressAsString(ip);
		port = endpoint.port;
	}
	void Send(const char* data, int size) {
#ifdef _WIN32
		SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
		if (sock == INVALID_SOCKET) { post("TCP send: socket failed"); return; }
#else
		int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
		if (sock < 0) { post("TCP send: socket failed"); return; }
#endif
		sockaddr_in addr = {};
		addr.sin_family = AF_INET;
		addr.sin_port = htons(port);
		inet_pton(AF_INET, ip, &addr.sin_addr);
		if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
			post("TCP send: connect failed to %s:%d", ip, port);
#ifdef _WIN32
			closesocket(sock);
#else
			close(sock);
#endif
			return;
		}
		uint32_t len_net = htonl((uint32_t)size);
		::send(sock, (const char*)&len_net, 4, 0);
		::send(sock, data, size, 0);
#ifdef _WIN32
		closesocket(sock);
#else
		close(sock);
#endif
	}
};

/* This following class is for controling threading and data flow to and from the python server */
/* we have this class because this is much more convinient to call its instances and call its functions this way */

class thread_control
{
	std::mutex mutex;
	std::condition_variable condVar;

public:
	thread_control()
	{}
	void notify() /* this will be used to notify threads that server have done processing */
	{
		condVar.notify_one();
	}
	void waitforit() /* this will be used to stop threads and wait for responces from server */
	{
		std::unique_lock<std::mutex> mlock(mutex);
		condVar.wait(mlock);
	}
	void lock() {
		mutex.lock();
	}
	void unlock() {
		mutex.unlock();
	}
};



// Data Structures
typedef struct _multi_track {
	t_object	ob;

	double percentage;           // To store the percentage value
	//int calculated_samples;      // To store the result of 163840 * percentage
	double pr_win_mul;


	int* bass_index;        // Index for bass buffer
	int* drums_index;       // Index for drums buffer
	int* guitar_index;      // Index for guitar buffer
	int* piano_index;       // Index for piano buffer


	//char filename[MAX_PATH_CHARS]; /* Keras Network name saved as .h5 file */

	int verbose_flag; /* defines if extention will or won't output execution times and other information */

#ifdef _WIN32
	SHELLEXECUTEINFO lpExecInfo; /* Open .h5 file handler */
#endif

	/* paralel thereads for osc listening and outputing */
	t_systhread		listener_thread;

	/* comunication ports with pythin server */
	int PORT_SENDER;
	int PORT_LISTENER;

	//bool* server_predicted;		/* flag if server has predicted */
	//bool* server_ready;			/* flag if server is running and ready to receive data */
	//bool* python_import_done;	/* flag if server iported chunk of data and is ready to receive next chunk */

	/* thread controls */
	thread_control* out_tread_control;		/* lock, unlock and notify routine for output thread */
	thread_control* server_control;			/* lock, unlock and notify routine for server start up thread */
	thread_control* python_import_control;	/* lock, unlock and notify routine for server's data importing thread */

	/* outlets */
	void* bass_outlet;     // Outlet for bass buffer
	void* drums_outlet;    // Outlet for drums buffer
	void* guitar_outlet;   // Outlet for guitar buffer
	void* piano_outlet;    // Outlet for piano buffer

	int package_size;  // Variable to store the current packet size

#ifdef _WIN32
	HANDLE server_process = NULL; // Store the server process handle
	HANDLE server_job;  // Add this field
#else
	pid_t server_pid = 0;   // macOS: child process PID
	pid_t server_pgid = 0;  // macOS: child process group ID
	int terminal_window_id = 0; // macOS: Terminal.app window ID for clean close
	int server_osc_port = 7000; // macOS: server port, used to kill by port on stop
#endif

	char command_str[1024]; // Add this field to store the command

	char client_ip[512] = { 0 };
	char server_ip[512] = { 0 };

	int predict_flags[4];  // 1 = predict (send), 0 = skip
	int read_flag[4];  // 1 = read (send), 0 = skip

	std::chrono::high_resolution_clock::time_point packet_test_start_time;  // For round-trip timing
	std::chrono::high_resolution_clock::time_point data_import_start_time;  // For data import timing
	std::chrono::high_resolution_clock::time_point data_import_end_time;    // For data import end timing

	bool auto_load_model_on_ready;  // Flag to auto-load model when server is ready

} t_multi_track;


// Prototypes
t_multi_track* multi_track_new(t_symbol* s, long argc, t_atom* argv);

void		multi_track_assist(t_multi_track* x, void* b, long m, long a, char* s);
void		multi_track_free(t_multi_track* x);

void		multi_track_load_model(t_multi_track* x);
void		multi_track_server(t_multi_track* x, long command);
void		multi_track_verbose(t_multi_track* x, long command);
void		multi_track_set_percentage(t_multi_track* x, double percentage);
void		multi_track_set_pr_win_mul(t_multi_track* x, double pr_win_mul);


void		multi_track_jit_matrix(t_multi_track* x, t_symbol* s, long argc, t_atom* argv);  // to read and send matrix directly

void		multi_track_OSC_time_to_predict_sender(t_multi_track* x);
void		multi_track_OSC_load_model(t_multi_track* x);
void		send_acknowledgment(t_multi_track* x);

// data sending
void		send_matrix_plane(char* matrix_data, int plane, long dim0, long dim1, long dimstride0, long dimstride1, const char* address, int port, const char* tag, long package_size);

void		* multi_track_OSC_listener(t_multi_track* x, int argc, char* argv[]);
void		multi_track_OSC_listen_thread(t_multi_track* x);

void		multi_track_set_packet_size(t_multi_track* x, long new_size);
void		multi_track_test_packet(t_multi_track* x);

void		multi_track_send_print(t_multi_track* x);
void		multi_track_send_reset(t_multi_track* x);

void		multi_track_set_command(t_multi_track* x, t_symbol* s, long argc, t_atom* argv);

void		get_public_ip(char* ip_buffer, size_t buffer_size);
void		multi_track_get_client_ip(t_multi_track* x);

void		multi_track_set_predict_instruments(t_multi_track* x, t_symbol* s, long argc, t_atom* argv);
void		multi_track_set_read_instruments(t_multi_track* x, t_symbol* s, long argc, t_atom* argv);
void		multi_track_send_predict_instruments(t_multi_track* x);

void		timestamp();






// Globals and Statics
static t_class* s_multi_track_class = NULL;

///**********************************************************************/

// Class Definition and Life Cycle

void ext_main(void* r)
{
	t_class* c;

	//c = class_new("multi_track", (method)multi_track_new, (method)dsp_free, sizeof(t_multi_track), (method)NULL, A_GIMME, 0L);
	c = class_new("multi_track", (method)multi_track_new, (method)multi_track_free, sizeof(t_multi_track), (method)NULL, A_GIMME, 0L);


	// Add message handler for 'jit_matrix'
	class_addmethod(c, (method)multi_track_jit_matrix, "jit_matrix", A_GIMME, 0);
	

	class_addmethod(c, (method)multi_track_assist, "assist", A_CANT, 0);

	class_addmethod(c, (method)multi_track_load_model, "load_model", 0);

	class_addmethod(c, (method)multi_track_server, "server", A_LONG, 0);
	class_addmethod(c, (method)multi_track_verbose, "verbose", A_LONG, 0);

	class_addmethod(c, (method)multi_track_set_percentage, "percentage", A_FLOAT, 0);
	class_addmethod(c, (method)multi_track_set_pr_win_mul, "pr_win_mul", A_FLOAT, 0);


	class_addmethod(c, (method)multi_track_set_packet_size, "packet_size", A_LONG, 0);
	class_addmethod(c, (method)multi_track_test_packet, "test_packet", 0);
	class_addmethod(c, (method)multi_track_send_print, "print", 0);
	class_addmethod(c, (method)multi_track_send_reset, "reset", 0);

	class_addmethod(c, (method)multi_track_set_command, "set_command", A_GIMME, 0);
	class_addmethod(c, (method)multi_track_get_client_ip, "get_client_ip", 0);

	class_addmethod(c, (method)multi_track_OSC_time_to_predict_sender, "predict", 0); // to manually precit

	class_addmethod(c, (method)multi_track_set_predict_instruments, "predict_instruments", A_GIMME, 0);
	class_addmethod(c, (method)multi_track_set_read_instruments, "read_instruments", A_GIMME, 0);
	

	/* attributes */
	CLASS_ATTR_LONG(c, "verb", 0, t_multi_track, verbose_flag);

	class_dspinit(c);
	class_register(CLASS_BOX, c);
	s_multi_track_class = c;
}


/***********************************************************************/
/***********************************************************************/
/********************** Initialisation *********************************/
/***********************************************************************/
/***********************************************************************/

t_multi_track* multi_track_new(t_symbol* s, long argc, t_atom* argv)
{
	t_multi_track* x = (t_multi_track*)object_alloc(s_multi_track_class);

	if (x) {

		x->percentage = 0.0;          // Initialize percentage
		//x->calculated_samples = 0;    // Initialize calculated samples
		x->pr_win_mul = 0.0;

		//inlet_new(x, NULL);

		/* initialising variables */

		/* generate 2 random number that will beconme port numbers */
		srand(time(NULL));		 
		x->PORT_SENDER = 7000; //rand() % 100000;	
		x->PORT_LISTENER = 8000; //rand() % 100000;

		//memset(x->filename, 0, MAX_PATH_CHARS); /* emptying network name variable for the begining */

		x->verbose_flag = 0;

		// Dynamically allocate the indices
		x->bass_index = new int(0);
		x->drums_index = new int(0);
		x->guitar_index = new int(0);
		x->piano_index = new int(0);
					
		//x->lpExecInfo = NULL;			/* no need to be initialised */
		
		x->out_tread_control = new thread_control;
		x->server_control = new thread_control;
		x->python_import_control = new thread_control;
	
		/* here we check if argument are given */
		long offset = attr_args_offset((short)argc, argv); /* this is number of arguments before attributes start with @-sign */

		// Create outlets for left and right channels
		x->piano_outlet = listout((t_object*)x);  // Rightmost outlet
		x->guitar_outlet = listout((t_object*)x);
		x->drums_outlet = listout((t_object*)x);
		x->bass_outlet = listout((t_object*)x);  // Leftmost outlet


		attr_args_process(x, argc, argv); /* this is attribute reader */

		/* Starting up OSC listener server and output thread */
		multi_track_OSC_listen_thread(x);   /* this function starts OSC listener inside sub-thread */

		x->package_size = 10240;  // Default packet size

		get_public_ip(x->client_ip, sizeof(x->client_ip));
		// strncpy(x->client_ip, x->client_ip, sizeof(x->client_ip) - 1);
		post("Public IP: %s", x->client_ip);

		for (int i = 0; i < 4; i++) {
			x->predict_flags[i] = 0;  // Default: predict non of the instruments
			x->read_flag[i] = 0;	// Default: read non of the instruments
		}

		x->auto_load_model_on_ready = false;  // Initialize auto-load flag

	}
	return x;

}


void multi_track_free(t_multi_track* x) {
	post("Freeing multi_track object and stopping processes...");

	/************** Stop Python server if running ****************/
#ifdef _WIN32
	if (x->server_process) {
		DWORD exit_code;
		if (GetExitCodeProcess(x->server_process, &exit_code) && exit_code == STILL_ACTIVE) {
			post("Stopping Python server...");

			// Terminate the entire process tree if managed by a job object
			if (x->server_job) {
				TerminateJobObject(x->server_job, 0);
				CloseHandle(x->server_job);
				x->server_job = NULL;
			}

			// Terminate the server process
			TerminateProcess(x->server_process, 0);
			CloseHandle(x->server_process);
			x->server_process = NULL;

			post("Python server stopped.");
		}
	}
#else
	if (x->server_pgid > 0) {
		post("Stopping Python server...");
		killpg(x->server_pgid, SIGTERM);
		waitpid(x->server_pid, NULL, WNOHANG);
		x->server_pid = 0;
		x->server_pgid = 0;
		post("Python server stopped.");
	}
#endif

	///************** Stop listener thread safely ****************/
	//if (x->listener_thread) {
	//	post("Waiting for listener thread to exit...");
	//	systhread_join(x->listener_thread, 0); // Wait for thread to finish
	//	//systhread_exit(x->listener_thread, 0);
	//	systhread_sleep(500);
	//	x->listener_thread = NULL;
	//	post("Listener thread stopped.");
	//}

	/************** Free allocated memory ****************/
	delete x->bass_index;
	delete x->drums_index;
	delete x->guitar_index;
	delete x->piano_index;

	delete x->out_tread_control;
	delete x->server_control;
	delete x->python_import_control;

	post("Memory freed. Object cleanup complete.");

}





/****** getting and sending data from matrix ********/

void multi_track_jit_matrix(t_multi_track* x, t_symbol* s, long argc, t_atom* argv) {
	// Start timing
	x->data_import_start_time = std::chrono::high_resolution_clock::now();

	if (x->verbose_flag) {
		post("========================================");
		post("Data import started...");
	}

	if (argc < 1 || atom_gettype(argv) != A_SYM) {
		object_error((t_object*)x, "Invalid input. Expected a matrix name.");
		return;
	}

	// Extract matrix name
	t_symbol* matrix_name = atom_getsym(argv);
	t_object* matrix = (t_object*)jit_object_findregistered(matrix_name);

	if (!matrix) {
		object_error((t_object*)x, "Matrix not found: %s", matrix_name->s_name);
		return;
	}

	// Lock the matrix for safe access
	long lock = (long)jit_object_method(matrix, gensym("lock"), 1);

	// Get matrix info
	t_jit_matrix_info info;
	jit_object_method(matrix, gensym("getinfo"), &info);

	// Ensure the matrix has 4 planes and is float32
	if (info.type != _jit_sym_float32 || info.planecount != 4) {
		object_error((t_object*)x, "Matrix must have 4 planes of type float32.");
		jit_object_method(matrix, gensym("lock"), lock);
		return;
	}

	// Print matrix dimensions for debugging
	post("Matrix dimensions: %ld x %ld (4 planes)", info.dim[0], info.dim[1]);

	// Access matrix data
	char* matrix_data;
	jit_object_method(matrix, gensym("getdata"), &matrix_data);

	if (!matrix_data) {
		object_error((t_object*)x, "Failed to access matrix data.");
		jit_object_method(matrix, gensym("lock"), lock);
		return;
	}

	// Plane tags for OSC messages
	const char* plane_tags[4] = { "/bass", "/drums", "/guitar", "/piano" };

	//// Send each plane concurrently
	//std::thread bass_thread(send_matrix_plane, matrix_data, 0, info.dim[0], info.dim[1], info.dimstride[0], info.dimstride[1], x->server_ip, x->PORT_SENDER, plane_tags[0], x->package_size);
	//std::thread drums_thread(send_matrix_plane, matrix_data, 1, info.dim[0], info.dim[1], info.dimstride[0], info.dimstride[1], x->server_ip, x->PORT_SENDER, plane_tags[1], x->package_size);
	//std::thread guitar_thread(send_matrix_plane, matrix_data, 2, info.dim[0], info.dim[1], info.dimstride[0], info.dimstride[1], x->server_ip, x->PORT_SENDER, plane_tags[2], x->package_size);
	//std::thread piano_thread(send_matrix_plane, matrix_data, 3, info.dim[0], info.dim[1], info.dimstride[0], info.dimstride[1], x->server_ip, x->PORT_SENDER, plane_tags[3], x->package_size);

	//bass_thread.join();
	//drums_thread.join();
	//guitar_thread.join();
	//piano_thread.join();

	// Create threads for sending only non-predicted instruments
	std::thread* threads[4] = { nullptr };

	for (int i = 0; i < 4; i++) {
		if (x->read_flag[i] == 1) {  // Send only if read falag is 1
			threads[i] = new std::thread(send_matrix_plane, matrix_data, i, info.dim[0], info.dim[1],
				info.dimstride[0], info.dimstride[1], x->server_ip, x->PORT_SENDER,
				plane_tags[i], x->package_size);
		}
		//else {
		//	post("Skipping %s (predicted by model)", plane_tags[i]);
		//}
	}

	// Join threads
	for (int i = 0; i < 4; i++) {
		if (threads[i]) {
			threads[i]->join();
			delete threads[i];
		}
	}

	// Unlock the matrix
	jit_object_method(matrix, gensym("lock"), lock);

	// End timing
	x->data_import_end_time = std::chrono::high_resolution_clock::now();
	double import_time_ms = std::chrono::duration<double, std::milli>(x->data_import_end_time - x->data_import_start_time).count();

	if (x->verbose_flag) {
		post("All planes sent successfully.");
		post("Data import completed in %.3f ms", import_time_ms);
		post("========================================");
	}

	// Reset all instrument indices
	*x->bass_index = 0;
	*x->drums_index = 0;
	*x->guitar_index = 0;
	*x->piano_index = 0;

	// Send triger to start prediction process
	multi_track_OSC_time_to_predict_sender(x);
}





/**********************************************************************/
/**********************************************************************/
/*************************** Methods ********************************/
/**********************************************************************/
/**********************************************************************/

void multi_track_set_packet_size(t_multi_track* x, long new_size) {
	if (new_size < 128 || new_size > 16384) {  // Limit range for safety
		post("Invalid packet size. Choose between 128 and 16384 bytes.");
		return;
	}

	x->package_size = new_size;
	post("Packet size set to %d", x->package_size);

	// Send updated package size (chunk size) to Python
	TcpTransmitSocket transmitSocket(IpEndpointName(x->server_ip, x->PORT_SENDER));

	char buffer[256];
	osc::OutboundPacketStream p(buffer, 256);

	p << osc::BeginMessage("/update_package_size")
		<< x->package_size  // Send the new package size to Python
		<< osc::EndMessage;

	transmitSocket.Send(p.Data(), p.Size());
	post("Sent OSC message: /update_package_size with size %d", x->package_size);
}

void multi_track_set_percentage(t_multi_track* x, double new_percentage) {
	if (new_percentage < 0.0 || new_percentage > 1.0) {  // Limit between 0 and 1
		post("Invalid percentage. Choose a value between 0.0 and 1.0.");
		return;
	}

	x->percentage = new_percentage;
	post("Percentage set to %f", x->percentage);

	// Send updated percentage to Python server
	TcpTransmitSocket transmitSocket(IpEndpointName(x->server_ip, x->PORT_SENDER));

	char buffer[256];
	osc::OutboundPacketStream p(buffer, 256);

	p << osc::BeginMessage("/update_percentage")
		<< x->percentage  // Send the new percentage to Python
		<< osc::EndMessage;

	transmitSocket.Send(p.Data(), p.Size());
	post("Sent OSC message: /update_percentage with value %f", x->percentage);
}


void multi_track_set_pr_win_mul(t_multi_track* x, double new_pr_win_mul) {
	if (new_pr_win_mul < 0.0 || new_pr_win_mul > 2.0) {  // Limit between 0 and 2
		post("Invalid percentage. Choose a value between 0.0 and 2.0.");
		return;
	}

	x->pr_win_mul = new_pr_win_mul;
	post("Percentage set to %f", x->pr_win_mul);

	// Send updated percentage to Python server
	TcpTransmitSocket transmitSocket(IpEndpointName(x->server_ip, x->PORT_SENDER));

	char buffer[256];
	osc::OutboundPacketStream p(buffer, 256);

	p << osc::BeginMessage("/pr_win_mul")
		<< x->pr_win_mul  // Send the new percentage to Python
		<< osc::EndMessage;

	transmitSocket.Send(p.Data(), p.Size());
	post("Sent OSC message: /pr_win_mul with value %f", x->pr_win_mul);
}


void multi_track_test_packet(t_multi_track* x) {
	// Only announce when verbose
	if (x->verbose_flag) {
		post("Starting packet test with size %d (floats)", x->package_size);
	}

	TcpTransmitSocket transmitSocket(IpEndpointName(x->server_ip, x->PORT_SENDER));
	char osc_buffer[OUTPUT_BUFFER_SIZE];

	// High-resolution timers
	auto t0 = std::chrono::high_resolution_clock::now();

	// Build OSC message
	osc::OutboundPacketStream p(osc_buffer, OUTPUT_BUFFER_SIZE);
	p.Clear();
	p << osc::BeginMessage("/packet_test") << x->package_size;

	// Generate random float data
	for (int i = 0; i < x->package_size; i++) {
		float rand_value = static_cast<float>(rand()) / RAND_MAX;
		p << rand_value;
	}
	p << osc::EndMessage;

	auto t1 = std::chrono::high_resolution_clock::now();

	// Send
	transmitSocket.Send(p.Data(), p.Size());

	auto t2 = std::chrono::high_resolution_clock::now();

	// Store the start time for round-trip measurement
	x->packet_test_start_time = t0;

	if (x->verbose_flag) {
		// Timings
		double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		double send_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

		// Sizes and rates
		const size_t bytes_sent = p.Size();
		const double mb_sent = static_cast<double>(bytes_sent) / (1024.0 * 1024.0);
		const double mbit_sent = static_cast<double>(bytes_sent) * 8.0 / 1e6;
		const double mbps = (send_ms > 0.0) ? (mbit_sent / (send_ms / 1000.0)) : 0.0;

		post("Packet test built in %.3f ms, sent in %.3f ms", build_ms, send_ms);
		post("Bytes sent: %zu (%.3f MB); Effective rate: %.3f Mbit/s", bytes_sent, mb_sent, mbps);
	}
}



void multi_track_set_predict_instruments(t_multi_track* x, t_symbol* s, long argc, t_atom* argv) {
	if (argc != 4) {
		post("Error: predict_instruments requires exactly 4 arguments (bass, drums, guitar, piano), received %ld", argc);
		return;
	}

	for (int i = 0; i < 4; i++) {
		if (atom_gettype(&argv[i]) == A_LONG) {
			int value = atom_getlong(&argv[i]);
			if (value == 1 || value == 0) {
				x->predict_flags[i] = value;
			}
			else {
				post("Error: Argument %d must be 1 (predict), 0 (not predict) received %d", i, value);
				return;
			}
		}
		else {
			post("Error: Argument %d is not an integer!", i);
			return;
		}
	}

	post("Prediction selection updated: bass=%d drums=%d guitar=%d piano=%d",
		x->predict_flags[0], x->predict_flags[1], x->predict_flags[2], x->predict_flags[3]);

	multi_track_send_predict_instruments(x);
}


void multi_track_set_read_instruments(t_multi_track* x, t_symbol* s, long argc, t_atom* argv) {
	if (argc != 4) {
		post("Error: predict_instruments requires exactly 4 arguments (bass, drums, guitar, piano), received %ld", argc);
		return;
	}

	for (int i = 3; i >= 0; i--) {
		if (atom_gettype(&argv[i]) == A_LONG) {
			int value = atom_getlong(&argv[i]);
			if (value == 1 || value == 0) {
				x->read_flag[i] = value;
			}
			else {
				post("Error: Argument %d must be 1 (predict), 0 (not predict) received %d", i, value);
				return;
			}
		}
		else {
			post("Error: Argument %d is not an integer!", i);
			return;
		}
	}

	post("Read instruments updated: bass=%d drums=%d guitar=%d piano=%d",
		x->read_flag[0], x->read_flag[1], x->read_flag[2], x->read_flag[3]);

	//multi_track_send_predict_instruments(x);
}


void multi_track_send_predict_instruments(t_multi_track* x) {

	// Send updated selection to Python server
	TcpTransmitSocket transmitSocket(IpEndpointName(x->server_ip, x->PORT_SENDER));
	char buffer[256];
	osc::OutboundPacketStream p(buffer, 256);

	p << osc::BeginMessage("/predict_instruments")
		<< x->predict_flags[0] << x->predict_flags[1]
		<< x->predict_flags[2] << x->predict_flags[3]
		<< osc::EndMessage;

	transmitSocket.Send(p.Data(), p.Size());
	post("Sent OSC message: /predict_instruments");
}




void multi_track_send_print(t_multi_track* x) {
	TcpTransmitSocket transmitSocket(IpEndpointName(x->server_ip, x->PORT_SENDER));

	char buffer[256];
	osc::OutboundPacketStream p(buffer, 256);

	p << osc::BeginMessage("/print") << true << osc::EndMessage;

	transmitSocket.Send(p.Data(), p.Size());
	post("Sent OSC message: /print 1");
}

void multi_track_send_reset(t_multi_track* x) {
	TcpTransmitSocket transmitSocket(IpEndpointName(x->server_ip, x->PORT_SENDER));

	char buffer[256];
	osc::OutboundPacketStream p(buffer, 256);

	p << osc::BeginMessage("/reset") << 1 << osc::EndMessage;

	transmitSocket.Send(p.Data(), p.Size());
	post("Sent OSC message: /reset");

	// Reset all instrument indices
	*x->bass_index = 0;
	*x->drums_index = 0;
	*x->guitar_index = 0;
	*x->piano_index = 0;

	post("Reset indexes");
}


void multi_track_set_command(t_multi_track* x, t_symbol* s, long argc, t_atom* argv) {
	if (argc < 1) {
		post("Usage: set_command <command> [--server_ip <IP>]");
		return;
	}

	// 1) Raw command from Max (quotes are already unescaped here)
	const char* raw = atom_getsym(argv)->s_name;
	strncpy(x->command_str, raw, sizeof(x->command_str) - 1);
	x->command_str[sizeof(x->command_str) - 1] = '\0';

	// 2) Default server_ip; parse from command if present
	strncpy(x->server_ip, "127.0.0.1", sizeof(x->server_ip) - 1);
	x->server_ip[sizeof(x->server_ip) - 1] = '\0';

	auto resolve_to_ip = [](const char* hostname, char* ip_out, size_t ip_out_size) {
		struct addrinfo hints = {}, *res = nullptr;
		hints.ai_family = AF_INET;
		hints.ai_socktype = SOCK_DGRAM;
		if (getaddrinfo(hostname, nullptr, &hints, &res) == 0 && res) {
			inet_ntop(AF_INET, &((struct sockaddr_in*)res->ai_addr)->sin_addr, ip_out, (socklen_t)ip_out_size);
			freeaddrinfo(res);
			return true;
		}
		return false;
	};

	if (char* flag = strstr(x->command_str, "--server_ip")) {
		// explicit --server_ip provided: parse and resolve it
		flag += (int)strlen("--server_ip");
		while (*flag == ' ') flag++;
		char* end = flag;
		while (*end && *end != ' ' && *end != '\"') end++;
		size_t len = (size_t)(end - flag);
		if (len > 0 && len < sizeof(x->server_ip)) {
			char hostname[512] = { 0 };
			strncpy(hostname, flag, len);
			hostname[len] = '\0';
			if (!resolve_to_ip(hostname, x->server_ip, sizeof(x->server_ip)))
				strncpy(x->server_ip, hostname, sizeof(x->server_ip) - 1);
			x->server_ip[sizeof(x->server_ip) - 1] = '\0';
		}
	}
	else if (strstr(x->command_str, " -L ")) {
		// -L tunnel present: Max connects to localhost, keep 127.0.0.1
		post("TCP tunnel detected (-L), using server_ip=127.0.0.1");
	}
	else if (char* ssh = strstr(x->command_str, "ssh ")) {
		// no --server_ip: try to extract hostname from ssh user@host
		char* at = strchr(ssh, '@');
		if (at) {
			at++; // skip '@'
			char* end = at;
			while (*end && *end != ' ' && *end != '\"') end++;
			size_t len = (size_t)(end - at);
			if (len > 0 && len < 512) {
				char hostname[512] = { 0 };
				strncpy(hostname, at, len);
				hostname[len] = '\0';
				if (!resolve_to_ip(hostname, x->server_ip, sizeof(x->server_ip)))
					strncpy(x->server_ip, hostname, sizeof(x->server_ip) - 1);
				x->server_ip[sizeof(x->server_ip) - 1] = '\0';
			}
		}
	}

	// 3) Refresh client IP (your get_public_ip now returns the public IP and logs both)
	get_public_ip(x->client_ip, sizeof(x->client_ip));
	post("Server IP set to: %s", x->server_ip);
	post("Client IP set automatically to: %s", x->client_ip);

	// 4) If --client_ip already present anywhere, keep the command as-is
	if (strstr(x->command_str, "--client_ip") != NULL) {
		post("Command set to: %s", x->command_str);
		return;
	}

	// 5) Find the right injection point and insert --client_ip
	// For local commands:  zsh -ic "... --clientport N"       → inject before last "
	// For SSH commands:    ssh ... "bash -ic '... --clientport N'"  → inject before last '
	// so the arg ends up inside the python command, not outside it.
	char* last_quote = strrchr(x->command_str, '\"');
	if (last_quote) {
		// Look for a single-quote between the second-to-last " and last_quote
		// (i.e. inside the outer double-quoted SSH block, after bash -ic ')
		char* inject_pos = last_quote; // default: before last "
		char* p = last_quote - 1;
		while (p > x->command_str) {
			if (*p == '\'') { inject_pos = p; break; } // found closing ' — inject here
			if (*p == '\"') break; // hit another ", stop looking
			--p;
		}

		// Build injection
		char inject[128];
		snprintf(inject, sizeof(inject), " --client_ip %s",
			x->client_ip[0] ? x->client_ip : "0.0.0.0");
		size_t inject_len = strlen(inject);
		size_t cmd_len = strlen(x->command_str);

		if (cmd_len + inject_len >= sizeof(x->command_str)) {
			post("Warning: command too long to inject --client_ip; skipping injection.");
			post("Command set to: %s", x->command_str);
			return;
		}

		size_t tail_len = cmd_len - (size_t)(inject_pos - x->command_str) + 1;
		memmove(inject_pos + inject_len, inject_pos, tail_len);
		memcpy(inject_pos, inject, inject_len);

		post("Command set to: %s", x->command_str);
		return;
	}

	// 6) Fallback: no double quotes found � append at end
	{
		char inject[128];
		snprintf(inject, sizeof(inject), " --client_ip %s",
			x->client_ip[0] ? x->client_ip : "0.0.0.0");
		if (strlen(x->command_str) + strlen(inject) < sizeof(x->command_str)) {
			strncat(x->command_str, inject, sizeof(x->command_str) - strlen(x->command_str) - 1);
			post("Command set to: %s", x->command_str);
		}
		else {
			post("Warning: command too long to append --client_ip; skipping injection.");
			post("Command set to: %s", x->command_str);
		}
	}
}






void get_public_ip(char* ip_buffer, size_t buffer_size) {
	if (!ip_buffer || buffer_size == 0) return;
	ip_buffer[0] = '\0';

#ifdef _WIN32
	// ---------- 1) Get outbound interface IP (local/NAT-side) ----------
	char outbound_ip[64] = { 0 };
	{
		WSADATA wsaData;
		if (WSAStartup(MAKEWORD(2, 2), &wsaData) == 0) {
			SOCKET sock = socket(AF_INET, SOCK_DGRAM, 0);
			if (sock != INVALID_SOCKET) {
				sockaddr_in server_addr = {};
				server_addr.sin_family = AF_INET;
				server_addr.sin_port = htons(80);
				inet_pton(AF_INET, "8.8.8.8", &server_addr.sin_addr);  // Google DNS

				if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) != SOCKET_ERROR) {
					sockaddr_in local_addr = {};
					int addr_len = sizeof(local_addr);
					if (getsockname(sock, (struct sockaddr*)&local_addr, &addr_len) == 0) {
						inet_ntop(AF_INET, &local_addr.sin_addr, outbound_ip, sizeof(outbound_ip));
					}
				}
				closesocket(sock);
			}
			WSACleanup();
		}
	}

	// ---------- 2) Fetch true public IP via HTTPS (api.ipify.org) ----------
	char public_ip[64] = { 0 };
	bool got_public = false;
	{
		HINTERNET hSession = WinHttpOpen(L"multi_track/1.0",
			WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
			WINHTTP_NO_PROXY_NAME,
			WINHTTP_NO_PROXY_BYPASS, 0);
		if (hSession) {
			HINTERNET hConnect = WinHttpConnect(hSession, L"api.ipify.org",
				INTERNET_DEFAULT_HTTPS_PORT, 0);
			if (hConnect) {
				HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", L"/",
					NULL, WINHTTP_NO_REFERER,
					WINHTTP_DEFAULT_ACCEPT_TYPES,
					WINHTTP_FLAG_SECURE);
				if (hRequest &&
					WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
						WINHTTP_NO_REQUEST_DATA, 0, 0, 0) &&
					WinHttpReceiveResponse(hRequest, NULL)) {

					DWORD avail = 0, read = 0;
					std::string body;
					do {
						if (!WinHttpQueryDataAvailable(hRequest, &avail) || avail == 0) break;
						std::string chunk(avail, '\0');
						if (!WinHttpReadData(hRequest, &chunk[0], avail, &read) || read == 0) break;
						chunk.resize(read);
						body += chunk;
					} while (avail > 0);

					if (!body.empty()) {
						while (!body.empty() && (body.back() == '\r' || body.back() == '\n' || body.back() == ' ' || body.back() == '\t'))
							body.pop_back();
						strncpy(public_ip, body.c_str(), sizeof(public_ip) - 1);
						public_ip[sizeof(public_ip) - 1] = '\0';
						got_public = true;
					}
				}
				if (hRequest) WinHttpCloseHandle(hRequest);
				WinHttpCloseHandle(hConnect);
			}
			WinHttpCloseHandle(hSession);
		}
	}

	// ---------- 3) Log both & choose return value ----------
	post("Outbound interface IP: %s", outbound_ip[0] ? outbound_ip : "(unknown)");
	if (got_public) {
		post("Public (NAT) IP: %s", public_ip);
		strncpy(ip_buffer, public_ip, buffer_size - 1);
		ip_buffer[buffer_size - 1] = '\0';
	}
	else {
		post("Public (NAT) IP: (failed to fetch via HTTPS)");
		if (outbound_ip[0]) {
			strncpy(ip_buffer, outbound_ip, buffer_size - 1);
			ip_buffer[buffer_size - 1] = '\0';
		}
	}

#else
	// ---------- macOS: 1) Get outbound interface IP via connect trick ----------
	char outbound_ip[64] = { 0 };
	{
		int sock = socket(AF_INET, SOCK_DGRAM, 0);
		if (sock >= 0) {
			struct sockaddr_in server_addr = {};
			server_addr.sin_family = AF_INET;
			server_addr.sin_port = htons(80);
			inet_pton(AF_INET, "8.8.8.8", &server_addr.sin_addr);

			if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == 0) {
				struct sockaddr_in local_addr = {};
				socklen_t addr_len = sizeof(local_addr);
				if (getsockname(sock, (struct sockaddr*)&local_addr, &addr_len) == 0) {
					inet_ntop(AF_INET, &local_addr.sin_addr, outbound_ip, sizeof(outbound_ip));
				}
			}
			close(sock);
		}
	}

	// ---------- macOS: 2) Fetch public IP via curl ----------
	char public_ip[64] = { 0 };
	bool got_public = false;
	{
		FILE* fp = popen("curl -s --max-time 5 https://api.ipify.org", "r");
		if (fp) {
			if (fgets(public_ip, sizeof(public_ip), fp)) {
				// trim whitespace
				size_t len = strlen(public_ip);
				while (len > 0 && (public_ip[len-1] == '\r' || public_ip[len-1] == '\n' || public_ip[len-1] == ' '))
					public_ip[--len] = '\0';
				if (len > 0) got_public = true;
			}
			pclose(fp);
		}
	}

	// ---------- 3) Log both & choose return value ----------
	post("Outbound interface IP: %s", outbound_ip[0] ? outbound_ip : "(unknown)");
	if (got_public) {
		post("Public (NAT) IP: %s", public_ip);
		strncpy(ip_buffer, public_ip, buffer_size - 1);
		ip_buffer[buffer_size - 1] = '\0';
	}
	else {
		post("Public (NAT) IP: (failed to fetch via curl)");
		if (outbound_ip[0]) {
			strncpy(ip_buffer, outbound_ip, buffer_size - 1);
			ip_buffer[buffer_size - 1] = '\0';
		}
	}
#endif
}


void multi_track_get_client_ip(t_multi_track* x) {
	get_public_ip(x->client_ip, sizeof(x->client_ip));
	post("Client public IP: %s", x->client_ip);
}




/* this function shows info when user brings mouse to inlets and outlets */
void  multi_track_assist(t_multi_track* x, void* b, long m, long a, char* s)
{
	if (m == ASSIST_INLET) { // inlet
		sprintf(s, "Mono sigal");
	}
	else {	// outlet
		switch (a) {
		case 0: sprintf(s, "Output Left"); break;
		case 1: sprintf(s, "Output Right"); break;
		}
	}
}

/* function that sets verbose flag. when werbose is 1 server outputs log data */
void multi_track_verbose(t_multi_track* x, long command) {

	x->verbose_flag = command;

	if (command == 1) {

		//post("Network file: %s", x->filename);
		post("Listening on port: %d", x->PORT_LISTENER);
		post("Sending to port: %d", x->PORT_SENDER);
		//post("Server running: %d", *x->server_ready);
	}
	else if (command == 0) {}

}



///********************** load network *******************/
void multi_track_load_model(t_multi_track* x) {
	post("Loading model...");

	// Call the existing function to send the package size, percentage and load model
	multi_track_set_packet_size(x, x->package_size);
	multi_track_set_percentage(x, x->percentage);
	multi_track_set_pr_win_mul(x, x->pr_win_mul);
	multi_track_send_predict_instruments(x);
	multi_track_OSC_load_model(x);

	post("Model load request sent.");
}


/****************************************************/
/******************* Python Server ******************/
/****************************************************/


#ifdef _WIN32
// Function to terminate a process and its children (Windows only)
void terminate_process_tree(DWORD process_id) {
	HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	if (hSnapshot == INVALID_HANDLE_VALUE) {
		post("Failed to create process snapshot.");
		return;
	}

	PROCESSENTRY32 pe;
	pe.dwSize = sizeof(PROCESSENTRY32);

	if (Process32First(hSnapshot, &pe)) {
		do {
			if (pe.th32ParentProcessID == process_id) {
				// Recursively terminate child processes
				terminate_process_tree(pe.th32ProcessID);
			}
		} while (Process32Next(hSnapshot, &pe));
	}

	// Terminate the main process
	HANDLE hProcess = OpenProcess(PROCESS_TERMINATE, FALSE, process_id);
	if (hProcess != NULL) {
		TerminateProcess(hProcess, 0);
		CloseHandle(hProcess);
	}

	CloseHandle(hSnapshot);
}
#else
// macOS: terminate process group (covers all child processes)
void terminate_process_tree(pid_t pgid) {
	if (pgid > 0)
		killpg(pgid, SIGTERM);
}
#endif


void multi_track_server(t_multi_track* x, long command) {
	if (command == 1) { // Start the server

#ifdef _WIN32
		if (x->server_process != NULL) {
			DWORD exit_code;
			if (GetExitCodeProcess(x->server_process, &exit_code) && exit_code == STILL_ACTIVE) {
				post("Server is already running.");
				return;
			}
		}
#else
		if (x->server_pid > 0 && kill(x->server_pid, 0) == 0) {
			post("Server is already running.");
			return;
		}
#endif

		if (x->command_str[0] == '\0') {
			post("Error: No command defined. Use 'set_command' to define the server command.");
			return;
		}

#ifdef _WIN32
		char command_str[512];
		snprintf(command_str, sizeof(command_str), "cmd.exe /C %s", x->command_str);

		STARTUPINFO si;
		PROCESS_INFORMATION pi;
		ZeroMemory(&si, sizeof(si));
		si.cb = sizeof(si);
		ZeroMemory(&pi, sizeof(pi));

		si.dwFlags = STARTF_USESHOWWINDOW;
		si.wShowWindow = SW_SHOWNORMAL;

		HANDLE hJob = CreateJobObject(NULL, NULL);
		if (hJob == NULL) {
			post("Failed to create job object.");
			return;
		}

		if (!CreateProcess(NULL, command_str, NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL, NULL, &si, &pi)) {
			post("Failed to start the server.");
			CloseHandle(hJob);
			return;
		}

		if (!AssignProcessToJobObject(hJob, pi.hProcess)) {
			post("Failed to assign process to job object.");
			CloseHandle(hJob);
			TerminateProcess(pi.hProcess, 0);
			CloseHandle(pi.hProcess);
			CloseHandle(pi.hThread);
			return;
		}

		x->server_process = pi.hProcess;
		x->server_job = hJob;
		CloseHandle(pi.hThread);

#else
		// macOS: write command to a temp shell script, then use osascript to
		// open it in Terminal.app and capture the window ID for clean shutdown.

		// Parse --serverport from command so stop can kill by port
		x->server_osc_port = 7000; // default
		if (char* pf = strstr(x->command_str, "--serverport")) {
			pf += strlen("--serverport");
			while (*pf == ' ') pf++;
			x->server_osc_port = atoi(pf);
		}

		// Write the server command to a temp shell script
		const char* sh_path = "/tmp/multi_track_server.sh";
		FILE* fp = fopen(sh_path, "w");
		if (fp) {
			fprintf(fp, "#!/bin/zsh\n%s\n", x->command_str);
			fclose(fp);
			chmod(sh_path, 0755);
		} else {
			post("Failed to write server launch script.");
			return;
		}

		// Write AppleScript to open the script in Terminal and return window ID
		const char* scpt_path = "/tmp/multi_track_open.scpt";
		FILE* scpt = fopen(scpt_path, "w");
		if (scpt) {
			fprintf(scpt, "tell application \"Terminal\"\n");
			fprintf(scpt, "  set newTab to do script \"%s\"\n", sh_path);
			fprintf(scpt, "  activate\n");
			fprintf(scpt, "  set winID to id of window 1\n");
			fprintf(scpt, "end tell\n");
			fprintf(scpt, "return winID\n");
			fclose(scpt);
		} else {
			post("Failed to write AppleScript.");
			return;
		}

		// Run osascript and capture the Terminal window ID
		char popen_cmd[256];
		snprintf(popen_cmd, sizeof(popen_cmd), "osascript %s", scpt_path);
		FILE* pipe = popen(popen_cmd, "r");
		if (pipe) {
			int win_id = 0;
			fscanf(pipe, "%d", &win_id);
			pclose(pipe);
			x->terminal_window_id = win_id;
			post("Server terminal window ID: %d", win_id);
		} else {
			post("Failed to launch Terminal via osascript.");
			return;
		}
#endif

		x->auto_load_model_on_ready = true;
		post("Server started. Waiting for server ready signal...");
	}
	else if (command == 0) { // Stop the server

#ifdef _WIN32
		if (x->server_process) {
			DWORD exit_code;
			if (GetExitCodeProcess(x->server_process, &exit_code)) {
				if (exit_code == STILL_ACTIVE) {
					post("Stopping the server...");
					if (x->server_job != NULL) {
						if (!TerminateJobObject(x->server_job, 0))
							post("Failed to terminate job object.");
						CloseHandle(x->server_job);
						x->server_job = NULL;
					}
					CloseHandle(x->server_process);
					x->server_process = NULL;
					post("Server stopped.");
				}
				else {
					post("Server was already stopped.");
					x->server_process = NULL;
				}
			}
			else {
				post("Failed to get exit code of the server process.");
			}
		}
		else {
			post("No active server process found.");
		}
#else
		post("Stopping the server...");
		// 1. Kill server process by port (reliable regardless of process hierarchy)
		char kill_cmd[256];
		snprintf(kill_cmd, sizeof(kill_cmd),
			"lsof -ti :%d | xargs kill -TERM 2>/dev/null; "
			"sleep 0.5; "
			"lsof -ti :%d | xargs kill -KILL 2>/dev/null; true",
			x->server_osc_port, x->server_osc_port);
		system(kill_cmd);

		// 2. Close the Terminal window by ID — no running process so no prompt
		if (x->terminal_window_id != 0) {
			char close_cmd[256];
			snprintf(close_cmd, sizeof(close_cmd),
				"osascript -e 'tell application \"Terminal\" to close (windows whose id is %d)'",
				x->terminal_window_id);
			system(close_cmd);
			x->terminal_window_id = 0;
		}
		x->server_pid = 0;
		x->server_pgid = 0;
		post("Server stopped.");
#endif
	}
}


/****************************************** Some help and debuging functions **************************/

/* expoerting memoruy to text file for testing */

void timestamp()     /*********this is used to track timing of data flow *****/
{
#ifdef _WIN32
	SYSTEMTIME t;
	GetSystemTime(&t);
	post("%02d:%02d.%4d\n", t.wMinute, t.wSecond, t.wMilliseconds);
#else
	struct timeval tv;
	gettimeofday(&tv, NULL);
	struct tm* tm_info = gmtime(&tv.tv_sec);
	post("%02d:%02d.%04d\n", tm_info->tm_min, tm_info->tm_sec, (int)(tv.tv_usec / 1000));
#endif
}
/**********************************************************************/


/**************************** OSC SERVER *****************************/



/*********************************************************************/
/*************************** Data Send *******************************/
/*********************************************************************/

/* function for data sending */

void send_matrix_plane(char* matrix_data, int plane, long dim0, long dim1, long dimstride0, long dimstride1, const char* address, int port, const char* tag, long package_size) {
	TcpTransmitSocket transmitSocket(IpEndpointName(address, port));
	char osc_buffer[OUTPUT_BUFFER_SIZE];

	// Divide the plane into chunks
	//int package_size = package_size; // 10240; // Maximum floats per OSC message
	int num_samples = dim0 * dim1; // Total samples in the plane
	int num_chunks = (num_samples + package_size - 1) / package_size; // Round up

	for (int chunk = 0; chunk < num_chunks; chunk++) {
		osc::OutboundPacketStream p(osc_buffer, OUTPUT_BUFFER_SIZE);
		p.Clear();
		p << osc::BeginMessage(tag);

		int start_index = chunk * package_size;
		p << start_index;
		int end_index = (start_index + package_size < num_samples) ? start_index + package_size : num_samples;

		for (int i = start_index; i < end_index; i++) {
			int row = i / dim1;
			int col = i % dim1;
			int matrix_index = row * dimstride0 / sizeof(float) + col * dimstride1 / sizeof(float) + plane;

			float value = ((float*)matrix_data)[matrix_index];
			p << value;
		}

		p << osc::EndMessage;
		transmitSocket.Send(p.Data(), p.Size());

		// Optional: Add a small delay to avoid overwhelming the receiver
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}
}



/* function to send network file name */
void multi_track_OSC_load_model(t_multi_track* x)
{

	TcpTransmitSocket transmitSocket(IpEndpointName(x->server_ip, x->PORT_SENDER));

	char buffer[4096];

	osc::OutboundPacketStream p(buffer, 4096);

	p << osc::BeginMessage("/load_model") << osc::EndMessage;


	transmitSocket.Send(p.Data(), p.Size());

}

/* send signal that server needs to predict now */
void multi_track_OSC_time_to_predict_sender(t_multi_track* x)
{
	int one = 1;

	TcpTransmitSocket transmitSocket(IpEndpointName(x->server_ip, x->PORT_SENDER));
	
	char buffer[32];

	osc::OutboundPacketStream p(buffer, 32);

	p << osc::BeginMessage("/predict") << one << osc::EndMessage;

	transmitSocket.Send(p.Data(), p.Size());

}

// Add a function to send acknowledgment
void send_acknowledgment(t_multi_track* x) {
	// Send the acknowledgment message to the Python server
	int one = 1;

	TcpTransmitSocket transmitSocket(IpEndpointName(x->server_ip, x->PORT_SENDER));

	char buffer[32];

	osc::OutboundPacketStream p(buffer, 32);

	p << osc::BeginMessage("/ack") << one << osc::EndMessage;

	transmitSocket.Send(p.Data(), p.Size());
}





/************************************************************/
/************************* Listening ************************/
/************************************************************/


class multi_track_packetListener : public osc::OscPacketListener {

public:

	/* these are thread controls that are defined here and will be pluged to look at thread controls from the object */
	thread_control* out_tread_control;
	thread_control* server_control;
	thread_control* python_import_control;

	/* allocating buffer for output data */
	void* bass_outlet;
	void* drums_outlet;
	void* guitar_outlet;
	void* piano_outlet;

	//int bass_index;        // Index for bass buffer
	//int drums_index;       // Index for drums buffer
	//int guitar_index;      // Index for guitar buffer
	//int piano_index;       // Index for piano buffer


	int* bass_index;
	int* drums_index;
	int* guitar_index;
	int* piano_index;

	//int* calculated_samples;

	std::chrono::high_resolution_clock::time_point* packet_test_start_time;  // Pointer to start time
	std::chrono::high_resolution_clock::time_point* data_import_start_time;  // Pointer to data import start time
	int* verbose_flag;  // Pointer to verbose flag
	bool* auto_load_model_on_ready;  // Pointer to auto-load flag
	t_multi_track* multi_track_obj;  // Pointer to main object for calling methods

	/* these are local variables that will be aslo read form object variables */
	//bool server_predicted=false;
	//bool server_ready = false;   ///////////////////////////////////////// this need to be fasle in the end verison
	//bool python_import_done = false;

	void ProcessBundle(const osc::ReceivedBundle& b,
		const IpEndpointName& remoteEndpoint)
	{
		////ignore bundle time tag for now

		(void)remoteEndpoint; // suppress unused parameter warning

		for (osc::ReceivedBundle::const_iterator i = b.ElementsBegin();
			i != b.ElementsEnd(); ++i) {
			if (i->IsBundle())
				ProcessBundle(osc::ReceivedBundle(*i), remoteEndpoint);
			else
				ProcessMessage(osc::ReceivedMessage(*i), remoteEndpoint);   //, NULL
		}
	}




	virtual void ProcessMessage(const osc::ReceivedMessage& m, const IpEndpointName& remoteEndpoint)
	{
		(void)remoteEndpoint; // suppress unused parameter warning
		
		try {


			// Print the address pattern of the incoming message
			//post("Received OSC message: %s", m.AddressPattern());



			///* here comes signal that sevrer did append part of the data and needs next part, while sending data to server */
			//if (std::strcmp(m.AddressPattern(), "/appended") == 0) {
			//	osc::ReceivedMessageArgumentStream args = m.ArgumentStream();

			//	/**** Here we read osc message with local variable. this is read from the main stuct also by pointer ***/
			//	python_import_control->lock();
			//	args >> python_import_done >> osc::EndMessage;
			//	python_import_control->unlock();

			//	/***** notify data sending function that is waiting for this ****/
			//	python_import_control->notify();
			//}

			/* Here comes signal that server is loaded and running */
			if (std::strcmp(m.AddressPattern(), "/ready") == 0) {
				osc::ReceivedMessageArgumentStream args = m.ArgumentStream();
				bool server_ready;
				args >> server_ready >> osc::EndMessage;

				if (server_ready) {
					post("Server is ready!");

					// Auto-load model if flag is set
					if (*auto_load_model_on_ready) {
						post("Auto-loading model...");
						multi_track_load_model(multi_track_obj);
						*auto_load_model_on_ready = false;  // Reset flag after loading
					}

					// Notify waiting processes
					server_control->notify();
				}
			}

			/* Here comes signal that server is done sending predicted data */
			if (std::strcmp(m.AddressPattern(), "/server_predicted") == 0) {
				osc::ReceivedMessageArgumentStream args = m.ArgumentStream();

				out_tread_control->lock();
				bool server_predicted;
				args >> server_predicted >> osc::EndMessage;
				out_tread_control->unlock();

				if (server_predicted) {
					// Reset all instrument indices
					*bass_index = 0;
					*drums_index = 0;
					*guitar_index = 0;
					*piano_index = 0;

					// Calculate total processing time
					auto receive_end_time = std::chrono::high_resolution_clock::now();
					double total_time_ms = std::chrono::duration<double, std::milli>(receive_end_time - *data_import_start_time).count();

					if (*verbose_flag) {
						post("========================================");
						post("Data receive completed");
						post("Total processing time: %.3f ms", total_time_ms);
						post("========================================");
					}

					// Notify waiting processes
					out_tread_control->notify();
				}
			}

			if (std::strcmp(m.AddressPattern(), "/packet_test_response") == 0) {
				// Calculate round-trip time
				auto t_received = std::chrono::high_resolution_clock::now();
				double round_trip_ms = std::chrono::duration<double, std::milli>(t_received - *packet_test_start_time).count();

				osc::ReceivedMessageArgumentStream args = m.ArgumentStream();
				osc::int32 received_size;
				args >> received_size;

				// Allocate a C-style array dynamically
				float* received_values = new float[received_size];
				int count = 0;
				float value;

				// Extract all float values received
				while (!args.Eos() && count < received_size) {
					args >> value;
					received_values[count] = value;
					count++;
				}

				// Print the results with round-trip time
				post("========================================");
				post("Packet test completed. Round-trip time: %.3f ms", round_trip_ms);
				post("Received %d floats.", received_size);

				// Verify if the number of values matches the expected size
				if (count == received_size) {
					post("Packet integrity verified. All values received.");
				}
				else {
					post("WARNING: Expected %d values but received %d.", received_size, count);
				}
				post("========================================");

				// Free allocated memory
				delete[] received_values;
			}

			/* here comes prediction data in chunks of 10240 */
			// ##################################################################
			// Handle data messages for "/bass", "/drums", "/guitar", "/piano"
			void* target_outlet = nullptr;
			int* current_index = nullptr;

			if (std::strcmp(m.AddressPattern(), "/bass") == 0) {
				target_outlet = bass_outlet;
				current_index = bass_index;
			}
			else if (std::strcmp(m.AddressPattern(), "/drums") == 0) {
				target_outlet = drums_outlet;
				current_index = drums_index;
			}
			else if (std::strcmp(m.AddressPattern(), "/guitar") == 0) {
				target_outlet = guitar_outlet;
				current_index = guitar_index;
			}
			else if (std::strcmp(m.AddressPattern(), "/piano") == 0) {
				target_outlet = piano_outlet;
				current_index = piano_index;
			}
			else {
				//post("Unknown address pattern: %s", m.AddressPattern());
				return;
			}

			osc::ReceivedMessageArgumentStream args = m.ArgumentStream();

			// Process the OSC message data
			t_atom out_atom[2]; // Pair: index and value
			float value;

			while (!args.Eos()) {
				args >> value;

				//// Check if the current index exceeds the maximum limit
				//if (*current_index + 1 >= *calculated_samples) {
				//	post("Reached maximum samples for %s.", m.AddressPattern());
				//	post(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>receive ended"); timestamp();
				//	*current_index = 0;
				//	break;
				//}

				// Set the index and value to atoms
				atom_setlong(&out_atom[0], *current_index + 1); // 1-based index
				atom_setfloat(&out_atom[1], value);

				// Output to the target outlet
				outlet_list(target_outlet, 0L, 2, out_atom);
				//outlet_anything(target_outlet, out_atom, 0, NIL);

				// Increment the index
				(*current_index)++;
			}


			// ##################################################################


		}
		catch (osc::Exception& e) {

			post("error while parsing message from server!");

		}
	}

};

class CustomTcpListener {
private:
#ifdef _WIN32
	SOCKET server_fd;
#else
	int server_fd;
#endif
	char buffer[MAX_OSC_PACKET_SIZE];
	osc::OscPacketListener* packetListener;

public:
	CustomTcpListener(int port, osc::OscPacketListener* listener)
		: packetListener(listener) {

#ifdef _WIN32
		WSADATA wsaData;
		if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
			throw std::runtime_error("WSAStartup failed");
		server_fd = INVALID_SOCKET;
#endif

		server_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

#ifdef _WIN32
		if (server_fd == INVALID_SOCKET) { WSACleanup(); throw std::runtime_error("Socket creation failed"); }
#else
		if (server_fd < 0) throw std::runtime_error("Socket creation failed");
#endif

		int reuse = 1;
#ifdef _WIN32
		setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse));
#else
		setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
#endif

		sockaddr_in addr = {};
		addr.sin_family = AF_INET;
		addr.sin_addr.s_addr = INADDR_ANY;
		addr.sin_port = htons(port);

#ifdef _WIN32
		if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) { closesocket(server_fd); WSACleanup(); throw std::runtime_error("Bind failed"); }
		if (listen(server_fd, 5) == SOCKET_ERROR) { closesocket(server_fd); WSACleanup(); throw std::runtime_error("Listen failed"); }
#else
		if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) { close(server_fd); throw std::runtime_error("Bind failed"); }
		if (listen(server_fd, 5) < 0) { close(server_fd); throw std::runtime_error("Listen failed"); }
#endif
		char bound_ip[INET_ADDRSTRLEN];
		inet_ntop(AF_INET, &addr.sin_addr, bound_ip, sizeof(bound_ip));
		post("TCP listener initialized on %s:%d", bound_ip, port);
	}

	~CustomTcpListener() {
#ifdef _WIN32
		closesocket(server_fd);
		WSACleanup();
#else
		close(server_fd);
#endif
	}

	void run() {
		post("TCP listener waiting for connections");
		while (true) {
			sockaddr_in clientAddr = {};
#ifdef _WIN32
			int clientAddrLen = sizeof(clientAddr);
			SOCKET client_fd = accept(server_fd, (struct sockaddr*)&clientAddr, &clientAddrLen);
			if (client_fd == INVALID_SOCKET) continue;
#else
			socklen_t clientAddrLen = sizeof(clientAddr);
			int client_fd = accept(server_fd, (struct sockaddr*)&clientAddr, &clientAddrLen);
			if (client_fd < 0) continue;
#endif
			post("TCP listener: connection accepted");
			// Read multiple messages from this connection until it closes
			while (true) {
				uint32_t len_net = 0;
				int received = 0;
				while (received < 4) {
					int r = recv(client_fd, ((char*)&len_net) + received, 4 - received, 0);
					if (r <= 0) goto close_client;
					received += r;
				}
				{
					uint32_t len = ntohl(len_net);
					if (len == 0 || len > MAX_OSC_PACKET_SIZE) goto close_client;
					received = 0;
					while ((uint32_t)received < len) {
						int r = recv(client_fd, buffer + received, (int)(len - received), 0);
						if (r <= 0) goto close_client;
						received += r;
					}
					try {
						packetListener->ProcessPacket(buffer, received, IpEndpointName(clientAddr.sin_addr.s_addr, ntohs(clientAddr.sin_port)));
					}
					catch (const osc::Exception& e) {
						std::cerr << "Error processing OSC packet: " << e.what() << std::endl;
					}
				}
			}
			close_client:
#ifdef _WIN32
			closesocket(client_fd);
#else
			close(client_fd);
#endif
		}
	}
};

/* starting listerenr server thread */
void multi_track_OSC_listen_thread(t_multi_track* x)
{
	// create new thread + begin execution
	if (x->listener_thread == NULL) {

		systhread_create((method)multi_track_OSC_listener, x, 0, 0, 0, &x->listener_thread);
	}
}


void* multi_track_OSC_listener(t_multi_track* x, int argc, char* argv[]) {
	(void)argc; // Suppress unused parameter warnings
	(void)argv; // Suppress unused parameter warnings

	try {
		multi_track_packetListener listener;

		// Map struct variables to listener's local variables
		//x->server_predicted = &listener.server_predicted;
		//x->server_ready = &listener.server_ready;
		//x->python_import_done = &listener.python_import_done;

		// Map listener variables to the corresponding struct variables
		listener.bass_outlet = x->bass_outlet;
		listener.drums_outlet = x->drums_outlet;
		listener.guitar_outlet = x->guitar_outlet;
		listener.piano_outlet = x->piano_outlet;


		listener.bass_index = x->bass_index;        // Index for bass buffer
		listener.drums_index = x->drums_index;      // Index for drums buffer
		listener.guitar_index = x->guitar_index;    // Index for guitar buffer
		listener.piano_index = x->piano_index;      // Index for piano buffer

		//listener.calculated_samples = &x->calculated_samples;

		listener.packet_test_start_time = &x->packet_test_start_time;  // Map packet test timing
		listener.data_import_start_time = &x->data_import_start_time;  // Map data import timing
		listener.verbose_flag = &x->verbose_flag;  // Map verbose flag
		listener.auto_load_model_on_ready = &x->auto_load_model_on_ready;  // Map auto-load flag
		listener.multi_track_obj = x;  // Map main object pointer

		listener.out_tread_control = x->out_tread_control;
		listener.server_control = x->server_control;
		listener.python_import_control = x->python_import_control;

		// Create and configure the custom listener
		CustomTcpListener tcpListener(x->PORT_LISTENER, &listener);

		// Start listening for OSC messages
		tcpListener.run();
	}
	catch (const std::exception& e) {
		post("Error starting OSC listener: %s", e.what());
	}

	return NULL;
}

