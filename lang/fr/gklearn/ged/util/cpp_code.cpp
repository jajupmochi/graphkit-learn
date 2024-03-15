		else if (option.first == "random-inits") {
			try {
				num_random_inits_ = std::stoul(option.second);
				desired_num_random_inits_ = num_random_inits_;
			}
			catch (...) {
				throw Error(std::string("Invalid argument \"") + option.second + "\" for option random-inits. Usage: options = \"[--random-inits <convertible to int greater 0>]\"");
			}
			if (num_random_inits_ <= 0) {
				throw Error(std::string("Invalid argument \"") + option.second + "\" for option random-inits. Usage: options = \"[--random-inits <convertible to int greater 0>]\"");
			}
		}
		else if (option.first == "randomness") {
			if (option.second == "PSEUDO") {
				use_real_randomness_ = false;
			}
			else if (option.second == "REAL") {
				use_real_randomness_ = true;
			}
			else {
				throw Error(std::string("Invalid argument \"") + option.second  + "\" for option randomness. Usage: options = \"[--randomness REAL|PSEUDO] [...]\"");
			}
		}
		else if (option.first == "stdout") {
			if (option.second == "0") {
				print_to_stdout_ = 0;
			}
			else if (option.second == "1") {
				print_to_stdout_ = 1;
			}
			else if (option.second == "2") {
				print_to_stdout_ = 2;
			}
			else {
				throw Error(std::string("Invalid argument \"") + option.second  + "\" for option stdout. Usage: options = \"[--stdout 0|1|2] [...]\"");
			}
		}
		else if (option.first == "refine") {
			if (option.second == "TRUE") {
				refine_ = true;
			}
			else if (option.second == "FALSE") {
				refine_ = false;
			}
			else {
				throw Error(std::string("Invalid argument \"") + option.second  + "\" for option refine. Usage: options = \"[--refine TRUE|FALSE] [...]\"");
			}
		}
		else if (option.first == "time-limit") {
			try {
				time_limit_in_sec_ = std::stod(option.second);
			}
			catch (...) {
				throw Error(std::string("Invalid argument \"") + option.second + "\" for option time-limit.  Usage: options = \"[--time-limit <convertible to double>] [...]");
			}
		}
		else if (option.first == "max-itrs") {
			try {
				max_itrs_ = std::stoi(option.second);
			}
			catch (...) {
				throw Error(std::string("Invalid argument \"") + option.second + "\" for option max-itrs. Usage: options = \"[--max-itrs <convertible to int>] [...]");
			}
		}
		else if (option.first == "max-itrs-without-update") {
			try {
				max_itrs_without_update_ = std::stoi(option.second);
			}
			catch (...) {
				throw Error(std::string("Invalid argument \"") + option.second + "\" for option max-itrs-without-update. Usage: options = \"[--max-itrs-without-update <convertible to int>] [...]");
			}
		}
		else if (option.first == "seed") {
			try {
				seed_ = std::stoul(option.second);
			}
			catch (...) {
				throw Error(std::string("Invalid argument \"") + option.second + "\" for option seed. Usage: options = \"[--seed <convertible to int greater equal 0>] [...]");
			}
		}
		else if (option.first == "epsilon") {
			try {
				epsilon_ = std::stod(option.second);
			}
			catch (...) {
				throw Error(std::string("Invalid argument \"") + option.second + "\" for option epsilon. Usage: options = \"[--epsilon <convertible to double greater 0>] [...]");
			}
			if (epsilon_ <= 0) {
				throw Error(std::string("Invalid argument \"") + option.second + "\" for option epsilon. Usage: options = \"[--epsilon <convertible to double greater 0>] [...]");
			}
		}
		else if (option.first == "inits-increase-order") {
			try {
				num_inits_increase_order_ = std::stoul(option.second);
			}
			catch (...) {
				throw Error(std::string("Invalid argument \"") + option.second + "\" for option inits-increase-order. Usage: options = \"[--inits-increase-order <convertible to int greater 0>]\"");
			}
			if (num_inits_increase_order_ <= 0) {
				throw Error(std::string("Invalid argument \"") + option.second + "\" for option inits-increase-order. Usage: options = \"[--inits-increase-order <convertible to int greater 0>]\"");
			}
		}
		else if (option.first == "init-type-increase-order") {
			init_type_increase_order_ = option.second;
			if (option.second != "CLUSTERS" and option.second != "K-MEANS++") {
				throw ged::Error(std::string("Invalid argument ") + option.second + " for option init-type-increase-order. Usage: options = \"[--init-type-increase-order CLUSTERS|K-MEANS++] [...]\"");
			}
		}
		else if (option.first == "max-itrs-increase-order") {
			try {
				max_itrs_increase_order_ = std::stoi(option.second);
			}
			catch (...) {
				throw Error(std::string("Invalid argument \"") + option.second + "\" for option max-itrs-increase-order. Usage: options = \"[--max-itrs-increase-order <convertible to int>] [...]");
			}
		}
		else {
			std::string valid_options("[--init-type <arg>] [--random-inits <arg>] [--randomness <arg>] [--seed <arg>] [--stdout <arg>] ");
			valid_options += "[--time-limit <arg>] [--max-itrs <arg>] [--epsilon <arg>] ";
			valid_options += "[--inits-increase-order <arg>] [--init-type-increase-order <arg>] [--max-itrs-increase-order <arg>]";
			throw Error(std::string("Invalid option \"") + option.first + "\". Usage: options = \"" + valid_options + "\"");
		}
