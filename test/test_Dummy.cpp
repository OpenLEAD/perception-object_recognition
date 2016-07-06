#include <boost/test/unit_test.hpp>

using namespace object_recognition;

BOOST_AUTO_TEST_CASE(it_should_not_crash_when_welcome_is_called)
{
    std::string cloud_name  = "models/milk_cartoon_all_small_clorox.pcd";
    
    ORPointCloud or_cloud;
    or_cloud.importCloud(cloud_name, false);

    std::cout << or_cloud.file_name << std::endl;
    std::cout << or_cloud.file_name_raw << std::endl;
    

    cloud_name  = "models/milk_cartoon_all_small_clorox";

    or_cloud.importCloud(cloud_name, false);

    BOOST_CHECK(or_cloud.file_name==cloud_name);
    

}
